DATA_DIR=/home/jqcao/projects/memory_transformer/LongMem/data/wikitext-103/bin
SAVE_DIR=/home/jqcao/projects/memory_transformer/LongMem/checkpoint/newgpt-5e-5-longmem

CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 \
    fairseq-eval-lm ${DATA_DIR} \
    --path ${SAVE_DIR}/checkpoint100.pt \
    --batch-size 10 \
    --tokens-per-sample 1024 \
    --context-window 512 \
    --data-no-shuffle \
    --model-overrides '{"gpt_model_path":None}' \
    --gen-subset test