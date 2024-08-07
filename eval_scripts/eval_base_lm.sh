DATA_DIR=/home/jqcao/projects/memory_transformer/LongMem/data/wikitext-103/bin
SAVE_DIR=/home/jqcao/projects/memory_transformer/LongMem/checkpoint/newgpt-5e-5

CUDA_VISIBLE_DEVICES=3 NCCL_P2P_DISABLE=1 \
    fairseq-eval-lm ${DATA_DIR} \
    --path ${SAVE_DIR}/checkpoint_best.pt \
    --batch-size 6 \
    --tokens-per-sample 1024 \
    --context-window 512 \
    --data-no-shuffle \
    --gen-subset test