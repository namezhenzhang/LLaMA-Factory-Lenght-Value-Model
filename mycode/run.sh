CUDA_VISIBLE_DEVICES=0,1 python mycode/batch_decode.py \
    --max-batch-size 350 \
    --num-questions 128 \
    --num-rollouts 4 \
    --repetition-penalty 1.0 \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --max-new-tokens 1500 \
    > mycode/output_baseline.txt

CUDA_VISIBLE_DEVICES=2,3 python mycode/batch_decode.py \
    --max-batch-size 256 \
    --max-lvm-batch-size 600 \
    --num-questions 128 \
    --num-rollouts 4 \
    --repetition-penalty 1.0 \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --max-new-tokens 1500 \
    --if-modify-probs \
    > mycode/output_modify_probs_shortest.txt

CUDA_VISIBLE_DEVICES=0,1 python mycode/batch_decode.py \
    --max-batch-size 256 \
    --max-lvm-batch-size 500 \
    --num-questions 128 \
    --num-rollouts 4 \
    --repetition-penalty 1.0 \
    --temperature 1.0 \
    --top-p 0.9 \
    --top-k 20 \
    --max-new-tokens 1800 \
    --if-modify-probs \
    > mycode/output_modify_probs_largest_0.9.txt

CUDA_VISIBLE_DEVICES=2,3 python mycode/batch_decode.py \
    --max-batch-size 256 \
    --max-lvm-batch-size 500 \
    --num-questions 128 \
    --num-rollouts 4 \
    --repetition-penalty 1.0 \
    --temperature 1.0 \
    --top-p 0.8 \
    --top-k 20 \
    --max-new-tokens 1900 \
    --if-modify-probs \
    > mycode/output_modify_probs_largest_0.8.txt


CUDA_VISIBLE_DEVICES=2,3 python mycode/batch_decode.py \
    --inference-model-dir Qwen/Qwen2.5-3B-Instruct \
    --length-value-model-dir saves/qwen2.5-3b/full/lvm-relative-dapo-17k-7398-16-lr2e-5-g0.997-l0.997-d0.5-gpu2-bs8-ga8-ep2-wu30-cut2500/checkpoint-50 \
    --max-batch-size 100 \
    --num-questions 8 \
    --num-rollouts 2 \
    --repetition-penalty 1.0 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --max-new-tokens 2048 \
    --if-modify-probs \
    --cpu-only \
    > mycode/output_modify_probs_largest.txt

