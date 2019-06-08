# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''


class Hyperparams:
    '''Hyperparameters'''
    # data
    source_vocab = 'preprocessed/src.vocab.tsv'
    target_vocab = 'preprocessed/tgt.vocab.tsv'
    source_train = 'corpora/train.src'
    target_train = 'corpora/train.tgt'
    source_test = 'corpora/test_cleaned.txt'


    # training  
    batch_size = 256  # alias = N
    warmup_steps = 3000
    lr = 0.0003  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    maxlen = 30  # Maximum number of words in a sentence. alias = T.
    offset = 6

    # Feel free to increase this if you are ambitious.
    min_cnt = 0  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 256  # alias = C
    num_blocks = 1  # number of encoder/decoder blocks
    num_epochs = 10
    num_heads = 4
    dropout_rate = 0.1
