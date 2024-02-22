DATA_DIR=/home/jqcao/projects/memory_transformer/LongMem/data/wikitext-103/bin
SAVE_DIR=/home/jqcao/projects/memory_transformer/LongMem/checkpoint/newgpt-5e-5
DSTROE_DIR=/home/jqcao/projects/memory_transformer/LongMem/key_value_dstore/5e-5_best

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 NCCL_P2P_DISABLE=1 \
    fairseq-eval-lm ${DATA_DIR} \
    --path ${SAVE_DIR}/checkpoint_best.pt \
    --batch-size 6 \
    --context-window 512 \
    --gen-subset train \
    --data-no-shuffle \
    --save-knnlm-dstore \
    --dstore-size 103226509 \
    --dstore-fp16 \
    --model-overrides '{"use_external_memory":True,"tokens_per_sample":1024}' \
    --dstore-mmap ${DSTROE_DIR} 