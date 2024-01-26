from fairseq.data import Dictionary
import os
from argparse import Namespace
from fairseq.data.encoders.gpt2_bpe import GPT2BPE

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"

project_path = "/home/jqcao/projects/memory_transformer/LongMem"
dict_path = os.path.join(project_path,"data/wikitext-103/bin/dict.txt")

dictionary = Dictionary.load(os.path.join(dict_path))
tokenizer =  GPT2BPE(Namespace(gpt2_vocab_bpe=DEFAULT_VOCAB_BPE, gpt2_encoder_json=DEFAULT_ENCODER_JSON))

string = "What's your name ?"
tokens = tokenizer.encode(string)
print(tokens)

tokenized_ids = [dictionary.bos()] + dictionary.encode_line(tokens, add_if_not_exist=False).tolist()
print(tokenized_ids)