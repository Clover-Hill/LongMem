import os 

project_path = "~/projects/memory_transformer/LongMem"
raw_path = os.path.join(project_path,"data/wikitext-103/raw")
token_path = os.path.join(project_path,"data/wikitext-103/token")
bin_path = os.path.join(project_path,"data/wikitext-103/bin-new")

os.system("python {}/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json {}/gpt2_bpe/encoder.json \
    --vocab-bpe {}/gpt2_bpe/vocab.bpe \
    --inputs {}/train.txt \
    --outputs {}/train.bpe \
    --keep-empty --workers 64".format(project_path,project_path, project_path, raw_path, token_path))

os.system("python {}/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json {}/gpt2_bpe/encoder.json \
    --vocab-bpe {}/gpt2_bpe/vocab.bpe \
    --inputs {}/test.txt \
    --outputs {}/test.bpe \
    --keep-empty --workers 64".format(project_path,project_path, project_path, raw_path, token_path))

os.system("python {}/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json {}/gpt2_bpe/encoder.json \
    --vocab-bpe {}/gpt2_bpe/vocab.bpe \
    --inputs {}/valid.txt \
    --outputs {}/valid.bpe \
    --keep-empty --workers 64".format(project_path,project_path, project_path, raw_path, token_path))

# add bos token to each line. fairseq uses newline \n as the eos token.
os.system("sed -i \"s/^/<s> &/g\" {}/test.bpe".format(token_path))
os.system("sed -i \"s/^/<s> &/g\" {}/train.bpe".format(token_path))
os.system("sed -i \"s/^/<s> &/g\" {}/valid.bpe".format(token_path))

# src_dict is only used to compress tokenization result to binary file
os.system("fairseq-preprocess --only-source --trainpref {}/train.bpe \
    --validpref {}/valid.bpe \
    --testpref {}/test.bpe \
    --destdir {} --workers 128".format(token_path, token_path, token_path, bin_path))