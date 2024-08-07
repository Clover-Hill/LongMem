WANDB_PROJECT="decouple-lm"
DATA_DIR=/home/jqcao/projects/memory_transformer/LongMem/data/wikitext-103/bin
CKPT_DIR=/home/jqcao/projects/memory_transformer/LongMem/checkpoint/newgpt-5e-5-longmem
PRETRAINED_DIR=/home/jqcao/projects/memory_transformer/LongMem/checkpoint/newgpt-5e-5

WANDB_NAME="newgpt-small-longmem" CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_P2P_DISABLE=1 \
fairseq-train ${DATA_DIR}  \
    --save-dir ${CKPT_DIR} \
    --task language_modeling --arch transformer_lm_sidenet_gpt2_small \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 \
    --lr 2e-4 --lr-scheduler polynomial_decay \
    --weight-decay 0.01 \
    --save-interval 1 --sample-break-mode none \
    --ddp-backend=no_c10d \
    --tokens-per-sample 1024 \
    --batch-size 8 --total-num-update 100000 --seed 42 \
    --pretrained-model-path ${PRETRAINED_DIR}/checkpoint_best.pt \
    --layer-reduction-factor 2 \
    --disable-validation \
    --use-external-memory --memory-size 65536 \
    --k 64 --chunk-size 4 \
    --fp16 \
    --use-gpu-to-search \
    --no-token-positional-embeddings \
    --data-no-shuffle \
    --retrieval-layer-index 9 \
    --reload-ptm-layer \
    --wandb-project $WANDB_PROJECT 

# The --pre-trained-model path refers to the path to reproduced GPT-2-Medium checkpoints. You can find the downloading Google Drive url in README.