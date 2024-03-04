DIR=/home/jqcao/projects/memory_transformer/LongMem

python ${DIR}/fairseq/build_dstore.py \
    --dstore_mmap ${DIR}/key_value_dstore/5e-5_best \
    --save_dir ${DIR}/key_value_dstore/lr_5e-5_chunk_4 \
    --dstore_size 103226509  \
    --n_head 12 \
    --dimension 768 \
    --num_keys_to_add_at_a_time 500000 \
    --dstore_fp16