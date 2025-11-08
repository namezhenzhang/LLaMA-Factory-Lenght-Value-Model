python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-3B-Instruct \
  --tp-size 1 \
  --dp-size 8 \
  --context-length 32768 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.85 \
  --port 10001

python /data1/zhenzhang/dir1/LLaMA-Factory-Lenght-Value-Model/data_preprocess/data_preprocess.py \
  --num-samples 10000 \
  --samples-per-question 8 \
  --max-concurrency 500 \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --temperature 0.6 \
  --max-tokens 8196 \
  --top-p 0.9 \
  --openai-base-url http://127.0.0.1:10001/v1 \
  --openai-api-key None \
  --save-path data/deepmath_10000_3b.json \
  --request-timeout 300

llamafactory-cli train examples/train_lvm/qwen3_lrm.yaml


curl http://127.0.0.1:10001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-noauth" \
  -d '{
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "messages": [{"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{{}}.\n\nCalculate the value of 1+1."}],
    "temperature": 0.6,
    "top_p": 0.9,
    "max_tokens": 8196,
    "stream": false
  }'