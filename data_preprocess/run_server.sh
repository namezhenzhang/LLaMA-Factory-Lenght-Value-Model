python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B-Instruct-2507 \
  --tp-size 2 \
  --dp-size 2 \
  --context-length 32768 \
  --host 0.0.0.0 \
  --mem-fraction-static 0.85 \
  --port 10001