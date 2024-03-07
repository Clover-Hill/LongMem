WANDB_PROJECT="decouple-lm"
DATA_DIR=/home/jqcao/projects/memory_transformer/LongMem/data/wikitext-103/bin
SAVE_DIR=/home/jqcao/projects/memory_transformer/LongMem/checkpoint/newgpt-5e-5-hybrid-test
DSTORE_DIR=/home/jqcao/projects/memory_transformer/LongMem/key_value_dstore/lr_5e-5_chunk_4
PRETRAINED_DIR=/home/jqcao/projects/memory_transformer/LongMem/checkpoint/newgpt-5e-5

WANDB_NAME="newgpt-small-hybrid" CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 \
    fairseq-train ${DATA_DIR} \
    --task language_modeling \
    --save-dir ${SAVE_DIR} \
    --arch newgpt-small \
    --newgpt-mode alibi \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 \
    --total-num-update 200000 --warmup-updates 1000 \
    --lr 5e-6 --lr-scheduler polynomial_decay \
    --weight-decay 0.01 \
    --log-interval 1 \
    --update-freq 4 \
    --save-interval-updates 5000 --sample-break-mode none \
    --tokens-per-sample 1024 \
    --batch-size 4 --seed 1 \
    --use-knn-memory \
    --gpt-model-path ${PRETRAINED_DIR}/checkpoint_best.pt \
    --retrieval-layer-index 9 \
    --probe 8 \
    --k 1024 \
    --dstore-size 103226509 \
    --dstore-dir ${DSTORE_DIR} \
    --dstore-fp16 
    # --wandb-project $WANDB_PROJECT 
