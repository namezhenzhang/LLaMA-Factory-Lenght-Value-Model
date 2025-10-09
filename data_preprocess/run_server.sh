python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --tp-size 1 \
  --dp-size 8 \
  --context-length 40000 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.85 \
  --port 10001