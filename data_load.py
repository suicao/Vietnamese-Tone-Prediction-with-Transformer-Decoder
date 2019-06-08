from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import re as regex
import re
from tqdm import tqdm
import math

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
stop_words = "\" \' [ ] . , ! : ; ?".split(" ")


def basic_tokenizer(sentence, lower=True):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w.lower() if lower else w for w in words if w != '' and w != ' ']


def load_source_vocab():
    return load_vocab(hp.source_vocab)


def load_target_vocab():
    return load_vocab(hp.target_vocab)


def load_vocab(path):
    vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(source_sents, target_sents):
    src2idx, idx2src = load_source_vocab()
    tgt2idx, idx2tgt = load_target_vocab()

    # Index
    x_list, y_list = [], []
    for source_sent, target_sent in tqdm(zip(source_sents, target_sents), desc="Preparing data: ",
                                         total=len(source_sents)):
        x = [src2idx.get(word, src2idx["<unk>"]) for word in basic_tokenizer(source_sent,lower=False)]
        y = [tgt2idx.get(word, tgt2idx["<unk>"]) for word in basic_tokenizer(target_sent)]
        if len(x) != len(y):
            print(x)
            continue
        if len(x) < 5:
            continue
        n_samples = math.ceil(len(x)/hp.maxlen)
        for i in range(n_samples):
            maybe_x = np.array(x[i*hp.maxlen:(i+1)*hp.maxlen])
            if np.sum(maybe_x > 8) < 5:
                continue
            x_list.append(maybe_x)
            y_list.append(np.array(y[i*hp.maxlen:(i+1)*hp.maxlen]))
    # Pad
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in tqdm(enumerate(zip(x_list, y_list)), desc="Padding: ", total=len(x_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))

    return X, Y


def create_test_data(source_sents):
    src2idx, idx2src = load_source_vocab()
    # Index
    x_list, sources, ids = [], [], []
    for source_sent in tqdm(source_sents, desc="Preparing data: ", total=len(source_sents)):
        source_sent = basic_tokenizer(source_sent,lower=False)
        x = [src2idx.get(word, src2idx["<unk>"]) for word in source_sent]
        x_list.append(np.array(x))
        sources.append(source_sent)
        ids.append(source_sent[0].strip())

    max_infer_len = np.max([len(x) for x in x_list])
    X = np.zeros([len(x_list), max_infer_len], np.int32)
    actual_lengths = []
    for i, x in tqdm(enumerate(x_list), desc="Padding: ", total=len(x_list)):
        actual_lengths.append(len(x))
        X[i] = np.lib.pad(x, [0, max_infer_len - len(x)], 'constant', constant_values=(0, 0))

    return X, sources, ids, actual_lengths


def load_train_data():
    src_train = hp.source_train
    tgt_train = hp.target_train

    src_sents = [line for line in
                 codecs.open(src_train, 'r', 'utf-8').read().split("\n") if line]
    tgt_sents = [line for line in
                 codecs.open(tgt_train, 'r', 'utf-8').read().split("\n") if line]

    X, Y = create_data(src_sents, tgt_sents)
    return X, Y


def load_test_data():
    src_sents = [line for line in
                 codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line]
    X, Sources, ids, actual_lengths = create_test_data(src_sents)
    return X, Sources, ids, np.asarray(actual_lengths)


def get_batch_data():
    # Load data
    X, Y = load_train_data()
    np.save("./preprocessed/X.npy", X)
    np.save("./preprocessed/Y.npy", Y)

 #   X = np.load("./preprocessed/X.npy")[:, :hp.maxlen]
 #   Y = np.load("./preprocessed/Y.npy")[:, :hp.maxlen]

    return X, Y

