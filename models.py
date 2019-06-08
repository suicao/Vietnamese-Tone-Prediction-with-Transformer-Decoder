from __future__ import print_function
from data_load import get_batch_data, load_source_vocab, load_target_vocab
from modules import *


class TransformerDecoder:
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.dropout = tf.placeholder(tf.bool, shape=())
            if is_training:
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            # Load vocabulary
            src2idx, idx2src = load_source_vocab()
            tgt2idx, idx2tgt = load_target_vocab()
            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec, self.lookup_table = embedding(self.x,
                                                        vocab_size=len(src2idx),
                                                        num_units=hp.hidden_units,
                                                        zero_pad=False,
                                                        pretrained=False,
                                                        of="src",
                                                        scope="src_embeddings")
                self.dec += positional_encoding(self.dec, hp.maxlen)

                ## Dropout
                self.dec = tf.layers.dropout(self.dec,
                                             rate=hp.dropout_rate,
                                             training=self.dropout)

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       values=self.dec,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       training=self.dropout,
                                                       causality=False,
                                                       scope="vanilla_attention")

                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       values=self.dec,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       training=self.dropout,
                                                       causality=False,
                                                       scope="vanilla_attention_2")

                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])
            # Final linear projection
            self.logits = tf.layers.dense(tf.reshape(self.dec, [-1, hp.hidden_units]), len(tgt2idx))
            self.logits = tf.reshape(self.logits, [-1, hp.maxlen, len(tgt2idx)])
            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))

            if is_training:
                self.istarget = tf.to_float(tf.greater(self.y, src2idx[":"]))
                self.acc = tf.reduce_sum(
                    tf.to_float(tf.equal(tf.reshape(self.preds, (-1, self.y.shape[1])), self.y)) * self.istarget) / (
                               tf.reduce_sum(self.istarget))
                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
                nonpadding = tf.to_float(tf.not_equal(self.y, tgt2idx["<pad>"]))  # 0: <pad>
                self.loss = tf.reduce_sum(tf.reshape(ce, (-1, self.y.shape[1])) * nonpadding) / (
                        tf.reduce_sum(nonpadding) + 1e-7)
                self.mean_loss = self.loss
                self.global_step = tf.train.get_or_create_global_step()
                self.lr = noam_scheme(hp.lr, self.global_step, hp.warmup_steps)
                self.optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
                self.train_op_noembeddings = self.optimizer.minimize \
                    (self.loss, global_step=self.global_step,
                     var_list=[var for var in tf.trainable_variables() if
                               ("src_embeddings" not in var.name and "tgt_embeddings" not in var.name)])
                tf.summary.scalar('lr', self.lr)
                tf.summary.scalar("loss", self.loss)
                tf.summary.scalar("global_step", self.global_step)

                self.summaries = tf.summary.merge_all()


