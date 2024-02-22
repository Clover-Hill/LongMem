DIR=/home/jqcao/projects/memory_transformer/LongMem

python ${DIR}/fairseq/build_dstore.py \
    --dstore_mmap ${DIR}/key_value_dstore/5e-5_best \
    --dstore_size 103226509  \
    --faiss_index ${DIR}/key_value_dstore/5e-5_best.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0