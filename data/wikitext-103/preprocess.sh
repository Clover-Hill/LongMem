TEXT=raw/
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/train.txt \
    --validpref $TEXT/valid.txt \
    --testpref $TEXT/test.txt \
    --destdir original \
    --workers 20