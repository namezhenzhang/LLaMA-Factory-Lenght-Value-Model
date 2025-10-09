# %%
# load dataset from zwhe99/DeepMath-103K

import asyncio
import json
import datasets
from tqdm.asyncio import tqdm_asyncio

MAX_CONCURRENCY = 1000
N_SAMPLES_PER_QUESTION = 3
NUM_SAMPLES = 20
SAVE_PATH = f"/data1/zhenzhang/dir1/LLaMA-Factory-Lenght-Value-Model/data/deepmath_{NUM_SAMPLES}.json"

dataset = datasets.load_dataset("zwhe99/DeepMath-103K")

print(dataset)

# %%
# random sample NUM_SAMPLES data from dataset
sampled_dataset = dataset['train'].shuffle(seed=42).select(range(NUM_SAMPLES))

# %%
# distribution of difficulty
import matplotlib.pyplot as plt
difficulty_distribution = list(sampled_dataset['difficulty'])

plt.hist(difficulty_distribution, bins=[i * 0.5 for i in range(int(min(difficulty_distribution) * 2), int(max(difficulty_distribution) * 2) + 2)], edgecolor='black')
plt.title('Distribution of Difficulty')
plt.xlabel('Difficulty')
plt.ylabel('Frequency')
plt.show()

# %%
#　'question', 'final_answer', 'difficulty', 'topic', 'r1_solution_1', 'r1_solution_2', 'r1_solution_3'
MATH_QUERY_TEMPLATE = """
Please reason step by step, and put your final answer within \\boxed{{}}.

{Question}
""".strip()
# Optional helper to inspect a single prompt
def build_prompt(question):
    return MATH_QUERY_TEMPLATE.format(Question=question)


def maybe_get(obj, key, default=None):
    if obj is None:
        return default
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default

# %%
import openai





async def generate_single_completion(client, sample, prompt):
    message = [{"role": "user", "content": prompt}]
    response = await client.chat.completions.create(
        model="Qwen/Qwen3-4B-Instruct-2507",
        messages=message,
        temperature=0.6,
        max_tokens=32768,
        top_p=0.9,
    )
    answer = response.choices[0].message.content
    usage = getattr(response, "usage", None)
    completion_tokens = maybe_get(usage, "completion_tokens")
    return {
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "value": answer
            },
        ],
        "meta_info": {
            "difficulty": sample["difficulty"],
            "answer_token_length": completion_tokens,
        },
    }


async def collect_dataset(sampled_dataset, n_samples):
    client = openai.AsyncClient(base_url=f"http://127.0.0.1:10001/v1", api_key="None")
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def generate_with_limit(sample, prompt):
        async with semaphore:
            return await generate_single_completion(client, sample, prompt)

    tasks = []
    for idx in range(len(sampled_dataset)):
        sample = sampled_dataset[idx]
        prompt = build_prompt(sample["question"])
        for _ in range(n_samples):
            tasks.append(asyncio.create_task(generate_with_limit(sample, prompt)))
    try:
        return await tqdm_asyncio.gather(*tasks, total=len(tasks), desc="Generating completions")
    finally:
        await client.close()


data = asyncio.run(collect_dataset(sampled_dataset, N_SAMPLES_PER_QUESTION))

# %%
with open(SAVE_PATH, "w") as f:
    json.dump(data, f, indent=4)

# %%

