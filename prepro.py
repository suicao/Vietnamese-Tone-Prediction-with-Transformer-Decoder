# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
from __future__ import print_function
from hyperparams import Hyperparams as hp
# import tensorflow as tf
# import numpy as np
import codecs
import os
import re
from collections import Counter
from tqdm import tqdm

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
stop_words = "\" \' [ ] . , ! : ; ?".split(" ")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
        # return [w.lower() for w in words if w not in stop_words and w != '' and w != ' ']
    return [w.lower() for w in words if w != '' and w != ' ']


def make_vocab(fpath, fname):
    text = codecs.open(fpath, 'r', encoding='utf-8').readlines()

    words = []
    for line in tqdm(text):
        words.extend(basic_tokenizer(line))

    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<pad>", "<unk>", "<s>", "</s>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == '__main__':
    make_vocab(hp.source_train, "src.vocab.tsv")
    make_vocab(hp.target_train, "tgt.vocab.tsv")
    print("Done")
