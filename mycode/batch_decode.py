# %%
import argparse

parser = argparse.ArgumentParser()

# 模型与设备相关参数
parser.add_argument(
    "--length-value-model-dir",
    type=str,
    default="/data2/zzhang/dir1/LLaMA-Factory-Lenght-Value-Model/saves/qwen2.5-3b/full/lvm-dapo-17k-112000-4-lr2e-5-g0.999-l0.997-d0.5-gpu4-bs2-ga64-ep5-wu30-cut4096",
)
parser.add_argument(
    "--inference-model-dir",
    type=str,
    default="Qwen/Qwen2.5-3B-Instruct",
)
parser.add_argument(
    "--use-gpu",
    dest="use_gpu",
    action="store_true",
    default=True,
    help="Use GPU if available (default: True).",
)
parser.add_argument(
    "--cpu-only",
    dest="use_gpu",
    action="store_false",
    help="Force using CPU even if GPU is available.",
)
parser.add_argument(
    "--enable-flash-attn",
    action="store_true",
    default=False,
    help="Enable flash attention if supported.",
)
parser.add_argument(
    "--if-modify-probs",
    dest="if_modify_probs",
    action="store_true",
    default=False,
    help="Enable LVM-based probability modification.",
)

# 采样与解码参数
parser.add_argument("--repetition-penalty", type=float, default=1.05)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top-p", type=float, default=0.8)
parser.add_argument("--top-k", type=int, default=20)
parser.add_argument("--max-new-tokens", type=int, default=2048)

# batch / rollout 参数
parser.add_argument("--num-questions", type=int, default=10)
parser.add_argument("--num-rollouts", type=int, default=8)
parser.add_argument("--max-batch-size", type=int, default=8)

args = parser.parse_args()
print(args)

length_value_model_dir = args.length_value_model_dir
inference_model_dir = args.inference_model_dir
use_gpu = args.use_gpu
enable_flash_attn = args.enable_flash_attn
if_modify_probs = args.if_modify_probs

repetition_penalty = args.repetition_penalty
temperature = args.temperature
top_p = args.top_p
top_k = args.top_k
max_new_tokens = args.max_new_tokens

max_batch_size = args.max_batch_size

num_questions = args.num_questions
num_rollouts = args.num_rollouts

# %%
import datasets
dataset = datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")
# print(dataset["train"][0])

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache
from copy import deepcopy
import torch
from tqdm.auto import tqdm
import re

if use_gpu and torch.cuda.is_available():
    if torch.cuda.device_count() >= 2:
        device_model = torch.device("cuda:0")
        device_lvm = torch.device("cuda:1")
    else:
        device_model = torch.device("cuda")
        device_lvm = device_model
else:
    device_model = torch.device("cpu")
    device_lvm = device_model

print("Using generation model device:", device_model)
print("Using LVM device:", device_lvm)

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(inference_model_dir)
# 使用左侧 padding 以便与自回归 KV cache 更好兼容
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    inference_model_dir,
    dtype="auto",
    attn_implementation="flash_attention_2" if enable_flash_attn else None
)
model.to(device_model)

print(model)

# %%
# Load Length Value Model
import os
import torch
from safetensors.torch import load_file as safe_load_file
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

try:
    from llamafactory.extras.constants import V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
except ImportError:
    V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"
    V_HEAD_WEIGHTS_NAME = "value_head.bin"


def load_value_head(model, model_dir: str):
    """Load the value head weights saved by fix_valuehead_checkpoint."""
    safe_path = os.path.join(model_dir, V_HEAD_SAFE_WEIGHTS_NAME)
    bin_path = os.path.join(model_dir, V_HEAD_WEIGHTS_NAME)

    if os.path.exists(safe_path):
        state_dict = safe_load_file(safe_path, device="cpu")
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError("Cannot find value head weights in {}".format(model_dir))

    # Prepend expected prefix so that load_state_dict() picks them up.
    state_dict = {f"{name}": tensor for name, tensor in state_dict.items()}
    print(state_dict)
    model.load_state_dict(state_dict, strict=False)

tokenizer_lvm = AutoTokenizer.from_pretrained(length_value_model_dir, trust_remote_code=True)
lvm_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    length_value_model_dir,
    trust_remote_code=True,
    dtype="auto",
    attn_implementation="flash_attention_2" if enable_flash_attn else None
)
lvm_model.to(device_lvm)

lvm_model.v_head.dropout = torch.nn.Identity()
# print(lvm_model.v_head.summary.weight)
load_value_head(lvm_model, length_value_model_dir)
# print(lvm_model.v_head.summary.weight)
lvm_model.eval()
print(lvm_model)

@torch.no_grad()
def modify_probs(
    probs: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    finished: torch.Tensor,
    lvm_past_key_values=None,
    initial_budget: torch.Tensor | None = None,
) -> torch.Tensor:
    """基于 LVM 的 KV cache，对候选 token 做单步前向。

    当前示例：只打印 value 相关指标，不修改 probs。
    额外传入 initial_budget，表示在生成开始前 LVM 预测的「剩余长度期望」（可作为全局 budget 使用）。
    """
    # 没有未结束样本，或者当前步还没有 LVM 的 KV cache，就直接返回
    if finished.all() or lvm_past_key_values is None:
        return probs

    # LVM 所在设备（可能和主模型不同）
    lvm_device = next(lvm_model.parameters()).device

    # 只对尚未结束的样本、且概率非零的位置进行处理
    active_mask = (~finished).unsqueeze(-1)  # (batch_size, 1)
    candidate_mask = (probs > 0) & active_mask  # (batch_size, vocab_size)

    nonzero_indices = torch.nonzero(candidate_mask, as_tuple=False)  # (num_candidates, 2)
    if nonzero_indices.numel() == 0:
        return probs

    # 后面要用 batch_indices 去索引 LVM 的 KV cache。
    # 对于新的 transformers Cache API，我们会用 Cache.batch_select_indices 来根据 batch_indices 生成子 cache。
    batch_indices = nonzero_indices[:, 0]  # 保持在 CPU，给 Cache.batch_select_indices 使用
    token_indices = nonzero_indices[:, 1]

    # 构造候选 token 输入：(num_candidates, 1)，放到 LVM 对应的设备上
    candidate_input_ids = token_indices.unsqueeze(-1).to(lvm_device)
    # 注意：这里不能用全 1 的 mask，否则 LVM 看不到完整历史，只会在“几乎无上下文”的情况下打分。
    # 应该复用当前步与主模型一致的 attention_mask，并按 batch_indices 选出对应样本。
    candidate_attention_mask = attention_mask[batch_indices].to(lvm_device)

    # 根据 transformers 版本，past_key_values 可能是 Cache 对象，也可能是旧版的 tuple 形式
    if isinstance(lvm_past_key_values, Cache):
        # 使用 Cache.batch_select_indices 按 batch_indices 选择对应 batch，并复制一份，避免修改原始 cache
        candidate_past = deepcopy(lvm_past_key_values)
        candidate_past.batch_select_indices(batch_indices.to("cpu"))
    else:
        # 兼容旧版 tuple[(key_states, value_states), ...] 形式
        expanded_past = []
        for key_states, value_states in lvm_past_key_values:
            key_states = key_states[batch_indices].to(lvm_device)
            value_states = value_states[batch_indices].to(lvm_device)
            expanded_past.append((key_states, value_states))
        candidate_past = tuple(expanded_past)

    # 利用 KV cache，只对候选 token 做单步前向，拿到 value
    with torch.no_grad():
        # AutoModelForCausalLMWithValueHead 的 forward 返回 (logits, loss, value)
        _, _, values = lvm_model(
            input_ids=candidate_input_ids,
            attention_mask=candidate_attention_mask,
            past_key_values=candidate_past,
            use_cache=True,
            return_past_key_values=False,
        )

    token_values = values[:, -1].to(torch.float32)  # (num_candidates,)
    sigmoid_val = torch.sigmoid(token_values)
    log_base = torch.log(torch.tensor(0.999, device=lvm_device, dtype=token_values.dtype))
    log_val = torch.log(1 - sigmoid_val) / log_base

    # 将 log_val 移到与 probs 相同的设备，方便后续处理
    log_val = log_val.to(probs.device)

    # 每个样本只保留「log_val 最大」的那个 token（期望剩余长度最长）
    batch_size, vocab_size = probs.shape
    best_val = torch.full((batch_size,), float("-inf"), device=probs.device)
    best_token = torch.zeros((batch_size,), dtype=torch.long, device=probs.device)

    for idx, (b_idx, t_idx) in enumerate(nonzero_indices):
        v = log_val[idx]
        if v > best_val[b_idx]:
            best_val[b_idx] = v
            best_token[b_idx] = t_idx

    new_probs = probs.clone()
    valid_batches = torch.isfinite(best_val)
    for b in range(batch_size):
        if not valid_batches[b] or finished[b]:
            continue
        new_probs[b] = 0.0
        new_probs[b, best_token[b]] = 1.0

    return new_probs

# %%
def top_p_filtering(logits: torch.Tensor, top_p: float = 1.0) -> torch.Tensor:
    """Top-p (nucleus) 采样过滤，返回过滤后的 logits。

    logits: (batch_size, vocab_size)，未归一化的 logits。
    top_p: 采样概率阈值，(0, 1]。top_p=1.0 等价于不过滤。
    """
    if top_p is None or top_p <= 0.0 or top_p >= 1.0:
        return logits

    # 对每个 batch 的 logits 排序（从大到小）
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 标记需要移除的 token：累积概率大于 top_p 的位置
    sorted_indices_to_remove = cumulative_probs > top_p

    # 保证每行至少保留一个 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # 把需要移除的位置 scatter 回原始顺序
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

    # 将这些位置的 logits 设为 -inf
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    return logits



def top_k_filtering(logits: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Top-k 采样过滤，返回过滤后的 logits。

    logits: (batch_size, vocab_size)，未归一化的 logits。
    top_k: 每个位置只保留概率最高的 top_k 个 token，<=0 时表示不过滤。
    """
    if top_k is None or top_k <= 0:
        return logits

    vocab_size = logits.size(-1)
    top_k = min(top_k, vocab_size)

    # 取得每行第 k 大的 logit，低于该阈值的全部置为 -inf
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_values = values[..., -1, None]
    logits = logits.masked_fill(logits < min_values, float('-inf'))
    return logits

@torch.no_grad()
def custom_generate_stepwise(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    top_p: float = 1.0,
    top_k: int = 0,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 512,
    prob_fn=modify_probs,
    eos_token_id=None,
    show_progress: bool = False,
):
    # 主生成模型所在设备
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=model_device)
    else:
        attention_mask = attention_mask.to(model_device)

    if eos_token_id is None:
        eos_token_id = model.generation_config.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_token_id = eos_token_id[0]

    batch_size = input_ids.size(0)
    generated = input_ids
    finished = torch.zeros(batch_size, dtype=torch.bool, device=model_device)

    # 第一次：整段 prompt forward，拿到 past_key_values 和第一步 logits
    outputs = model(
        input_ids=generated,
        attention_mask=attention_mask,
        use_cache=True,
    )
    logits = outputs.logits[:, -1, :]  # (batch, vocab)
    past_key_values = outputs.past_key_values

    # 同步初始化 LVM 的 KV cache，后续在 modify_probs 中使用
    lvm_past_key_values = None
    initial_budget = None  # 记录每个样本在生成开始前 LVM 预测的「剩余长度期望」
    if prob_fn is not None:
        # LVM 在 device_lvm 上运行，需要将输入迁移过去
        lvm_generated = generated.to(device_lvm)
        lvm_attention_mask = attention_mask.to(device_lvm)
        # AutoModelForCausalLMWithValueHead 的 forward 返回 (logits, loss, value, past_key_values)
        # 当设置 return_past_key_values=True 时才会返回第 4 个元素
        _, _, values, lvm_past_key_values = lvm_model(
            input_ids=lvm_generated,
            attention_mask=lvm_attention_mask,
            # 显式打开 use_cache，底层 CausalLM 才会返回 past_key_values
            use_cache=True,
            return_past_key_values=True,
        )
        print(lvm_past_key_values)
        # values: (batch_size, seq_len)，取每个样本最后一个位置的 value，作为「当前 token 之后剩余长度的期望」
        last_values = values[:, -1].to(torch.float32)  # (batch_size,)
        sigmoid_val = torch.sigmoid(last_values)
        log_base = torch.log(torch.tensor(0.999, device=last_values.device, dtype=last_values.dtype))
        # 这里沿用你在下方分析代码中的变换，将 value 映射为期望剩余长度
        initial_budget = torch.log(1 - sigmoid_val) / log_base  # (batch_size,)

    pbar = None
    if show_progress:
        pbar = tqdm(total=max_new_tokens, desc="Generating tokens", leave=True)

    for step in range(max_new_tokens):
        # 1) 在 logits 空间做温度缩放 + top-k + top-p 过滤
        logits_step = logits / max(temperature, 1e-8)

        # 应用 repetition_penalty：惩罚已经在当前序列中出现过的 token
        if repetition_penalty is not None and repetition_penalty > 1.0:
            with torch.no_grad():
                for b in range(batch_size):
                    if finished[b]:
                        continue
                    seen_tokens = torch.unique(generated[b])
                    token_logits = logits_step[b, seen_tokens]
                    penalized = torch.where(
                        token_logits > 0,
                        token_logits / repetition_penalty,
                        token_logits * repetition_penalty,
                    )
                    logits_step[b, seen_tokens] = penalized

        logits_step = top_k_filtering(logits_step, top_k=top_k)
        logits_step = top_p_filtering(logits_step, top_p=top_p)

        # 2) logits -> 概率
        probs = torch.softmax(logits_step, dim=-1)

        # 3) 用户在概率空间里改动（这里把 LVM 的 KV cache 也传进去）
        if prob_fn is not None:
            probs = prob_fn(
                probs,
                input_ids,
                attention_mask,
                finished,
                lvm_past_key_values,
                initial_budget,
            )

        # 4) 为安全起见再归一化一次
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # 5) 采样下一个 token
        next_tokens = torch.multinomial(probs, num_samples=1)  # (batch, 1)

        # 已经结束的序列继续生成 eos
        next_tokens = torch.where(
            finished.unsqueeze(-1),
            torch.full_like(next_tokens, eos_token_id),
            next_tokens,
        )

        # 5) 拼接到已有序列
        generated = torch.cat([generated, next_tokens], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_tokens, device=model_device)],
            dim=-1,
        )

        # 6) 更新 finished 标记
        finished = finished | (next_tokens.squeeze(-1) == eos_token_id)
        if finished.all():
            break

        # 7) 只用最后一个 token + KV cache 做下一步 forward
        outputs = model(
            input_ids=next_tokens,
            attention_mask=attention_mask,
            use_cache=True,
            past_key_values=past_key_values,
        )
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        # 同步更新 LVM 的 KV cache，保持与 generated/attention_mask 对齐
        if prob_fn is not None and lvm_past_key_values is not None:
            # 将本步的 token 和 mask 迁移到 LVM 设备上，保持与 LVM KV cache 一致
            lvm_next_tokens = next_tokens.to(device_lvm)
            lvm_attention_mask = attention_mask.to(device_lvm)
            _, _, _, lvm_past_key_values = lvm_model(
                input_ids=lvm_next_tokens,
                attention_mask=lvm_attention_mask,
                past_key_values=lvm_past_key_values,
                # 同样需要显式 use_cache=True 才会继续返回新的 past_key_values
                use_cache=True,
                return_past_key_values=True,
            )

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    return generated

# %%
prompts = []
answers = []
question_indices = []  # 记录每条样本对应的题目索引，方便后续打印

for q_idx in range(num_questions):
    prompt = dataset["train"][q_idx]["prompt"][0]["content"]
    answer = dataset["train"][q_idx]["reward_model"]["ground_truth"]
    for _ in range(num_rollouts):
        prompts.append(prompt)
        answers.append(answer)
        question_indices.append(q_idx)

print(f"Total samples: {len(prompts)} (questions={num_questions}, rollouts_per_question={num_rollouts})")

# messages_list = [
#     [{"role": "user", "content": prompt}] for prompt in prompts
# ]

messages_list = [
    [
        {
            "role": "user",
            "content": prompt,
        }
    ]
    for prompt in prompts
]

texts = [
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    for messages in messages_list
]
total_samples = len(texts)
outputs = [None] * total_samples

# 按 max_batch_size 分批推理，避免一次性 OOM
for start in range(0, total_samples, max_batch_size):
    end = min(start + max_batch_size, total_samples)

    batch_texts = texts[start:end]

    # left padding 已在 tokenizer 上通过 padding_side="left" 设置
    model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(model.device)

    # 使用逐 token、自定义概率修改的方式做 batch 解码
    generated_ids = custom_generate_stepwise(
        model,
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        prob_fn=modify_probs if if_modify_probs else None,
        show_progress=True,
    )

    # remove the input part and decode each sample
    input_lengths = [len(ids) for ids in model_inputs["input_ids"]]
    for i, gen_ids in enumerate(generated_ids):
        global_idx = start + i
        output_ids = gen_ids[input_lengths[i]:].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True)
        outputs[global_idx] = content

# 统计平均长度和正确率（按 “Answer: $5$” 这种格式匹配）
def extract_answer_in_tex_format(text: str) -> str | None:
    """从字符串中提取形如 'Answer: $...$' 的答案主体."""
    m = re.search(r"Answer:\s*\$(.+?)\$", text)
    if m:
        return m.group(1).strip()
    return None


total_length = 0
correct_count = 0
num_samples = len(outputs)

for i, (q_idx, prompt, content, answer) in enumerate(zip(question_indices, prompts, outputs, answers)):
    # 使用 tokenizer 计算输出的 token 长度
    length_i = len(tokenizer.encode(content, add_special_tokens=False))
    total_length += length_i

    # 提取标准答案和模型答案中 “Answer: $...$” 里的部分并比较
    gt_inner = str(answer)
    pred_inner = extract_answer_in_tex_format(content)
    is_correct = (gt_inner is not None) and (pred_inner is not None) and (gt_inner == pred_inner)
    correct_count += int(is_correct)

    rollout_idx = i % num_rollouts
    print(f"===== Question {q_idx} - Rollout {rollout_idx} =====")
    print("Prompt:\n", prompt)
    print("Ground Truth Answer:\n", answer)
    # print("Generated Content:\n", content)
    print(f"Output length (tokens): {length_i}, Correct: {is_correct}, GT_inner: {gt_inner}, Pred_inner: {pred_inner}")
    print()

if num_samples > 0:
    avg_length = total_length / num_samples
    accuracy = correct_count / num_samples
    print(f"Average output length (tokens): {avg_length:.2f}")
    print(f"Accuracy (Answer: $...$ match): {accuracy:.4f}")


# # %%
# attention_mask = (generated_ids != tokenizer.pad_token_id) & (generated_ids != tokenizer.eos_token_id)

# # 将生成结果和 mask 移动到 LVM 所在设备，再送入 LVM 计算 value
# lvm_device = next(lvm_model.parameters()).device
# generated_ids_lvm = generated_ids.to(lvm_device)
# attention_mask_lvm = attention_mask.to(lvm_device)

# with torch.no_grad():
#     # AutoModelForCausalLMWithValueHead: forward 返回 (logits, loss, value)
#     _, _, values = lvm_model(
#         input_ids=generated_ids_lvm,
#         attention_mask=attention_mask_lvm,
#         return_past_key_values=False,
#     )

# value_preds = values[0].to(torch.float32).cpu()
# # print(value_preds)


# decoded_tokens = tokenizer.convert_ids_to_tokens(generated_ids[0])
# print("\n=== Token Value Predictions ===")
# i = 0
# log_vals = []
# remaining_lengths = []
# for token, val in zip(decoded_tokens, value_preds.tolist(),strict=True):
#     remaining_length = len(decoded_tokens) - i -1
#     sigmoid_val = torch.sigmoid(torch.tensor(val))
#     log_val = torch.log(1-sigmoid_val) / torch.log(torch.tensor(0.999))
#     log_vals.append(log_val.item())
#     remaining_lengths.append(remaining_length)
#     print(f"{i}: {repr(tokenizer.convert_tokens_to_string([token])):>10s} : {log_val:.4f}, {remaining_length} ")
#     i +=1
