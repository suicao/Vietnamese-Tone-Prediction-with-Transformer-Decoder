from __future__ import print_function
from data_load import basic_tokenizer
import codecs
import os
from data_load import load_test_data, load_source_vocab, load_target_vocab
from models import *
from tqdm import tqdm
import math

def eval():
    # Load graph
    g = TransformerDecoder(is_training=False)
    print("Graph loaded")

    # Load data
    X, sources, ids, actual_lengths = load_test_data()

    sorted_lengths = np.argsort(actual_lengths)
    X = X[sorted_lengths]
    print(X.shape)

    src2idx, idx2src = load_source_vocab()
    tgt2idx, idx2tgt = load_target_vocab()

    # Start session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(graph=g.graph, config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
        print("Restored!")

        ## Inference
        if not os.path.exists('results'):
            os.mkdir('results')
        with codecs.open("results/{}.txt".format(hp.logdir), "w", "utf-8") as fout:
            batch_size = hp.batch_size
            num_batches = math.ceil(len(X)/batch_size)
            Y_preds = np.zeros_like(X) + 2

            for i in tqdm(range(num_batches), desc="Inference: "):
                indices = np.arange(i*batch_size, min((i+1)*batch_size, len(X)))
                step = 0
                max_steps = math.ceil((np.max(actual_lengths[indices]) - hp.offset)/(hp.maxlen - hp.offset))
                for step in range(max_steps):
                    end = min(step*(hp.maxlen - hp.offset) + hp.maxlen, X.shape[1])
                    start = end - hp.maxlen

                    x = X[indices, start: end]
                    _preds = sess.run(g.preds, {g.x: x, g.dropout: False})
                    if step > 0:
                        Y_preds[indices, start+hp.offset//2:end] = _preds[:, hp.offset//2:]
                    else:
                        Y_preds[indices, start:end] = _preds
 
            Y_preds = Y_preds[np.argsort(sorted_lengths)]
            for sent_id, source, preds, actual_length in zip(ids, sources, Y_preds, actual_lengths):
                formatted_pred = [idx2tgt[idx] if src2idx.get(source[id],1) > 8 else source[id] for id, idx in enumerate(preds[:actual_length])]
                formatted_pred[0] = sent_id
                fout.write(" ".join(formatted_pred) + "\n")


if __name__ == '__main__':
    eval()
    print("Done")
