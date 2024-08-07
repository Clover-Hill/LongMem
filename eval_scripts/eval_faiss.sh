DATA_DIR=/home/jqcao/projects/memory_transformer/LongMem/data/wikitext-103/bin
SAVE_DIR=/home/jqcao/projects/memory_transformer/LongMem/checkpoint/newgpt-5e-5-hybrid

CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 \
    fairseq-eval-lm ${DATA_DIR} \
    --path ${SAVE_DIR}/checkpoint1.pt \
    --batch-size 6 \
    --context-window 512 \
    --data-no-shuffle \
    --gen-subset test \
    --model-overrides '{"tokens_per_sample":1024, "use_knn_memory":True, "probe":8, "k":1024, "dstore_size": 103226509, "chunk_size":4 , "dstore_dir": "/home/jqcao/projects/memory_transformer/LongMem/key_value_dstore/lr_5e-5_chunk_4", "dstore_fp16":True, "faiss_index_mmap":False, "gpt_model_path":None}'