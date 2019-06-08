from __future__ import print_function
from models import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    g = TransformerDecoder(is_training=True)
    print("Graph loaded")
    X, Y = get_batch_data()

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    num_batch_train = len(X_train) // hp.batch_size
    num_batch_val = len(X_val) // hp.batch_size
    with tf.Session(graph=g.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if tf.train.latest_checkpoint(hp.logdir) is not None:
            saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Loaded parameter from {}".format(tf.train.latest_checkpoint(hp.logdir)))

        for epoch in range(1, hp.num_epochs + 1):
            train_op = g.train_op
            pbar = tqdm(range(num_batch_train), total=num_batch_train, ncols=170, leave=False, unit='b',
                        desc="Epoch {}, loss = inf".format(epoch))
            for step in pbar:
                x_batch = X_train[step * hp.batch_size:(step + 1) * hp.batch_size, :]
                y_batch = Y_train[step * hp.batch_size:(step + 1) * hp.batch_size, :]
                _, loss, acc = sess.run([train_op, g.mean_loss, g.acc], feed_dict={g.x: x_batch, g.y: y_batch, g.dropout:True})
                pbar.set_description(' Epoch {}, loss = {:.4f}, acc = {:.4f}'.format(epoch, loss, acc))

            pbar = tqdm(range(num_batch_val), total=num_batch_val, ncols=170, leave=False, unit='b',
                        desc="Validating: ".format(epoch))

            total_acc = 0
            for step in pbar:
                x_batch = X_val[step * hp.batch_size:(step + 1) * hp.batch_size, :]
                y_batch = Y_val[step * hp.batch_size:(step + 1) * hp.batch_size, :]
                [acc] = sess.run([g.acc], feed_dict={g.x: x_batch, g.y: y_batch,g.dropout:False})
                total_acc += acc
            print("Epoch {}, acc = {:.4f}".format(epoch, total_acc/num_batch_val))

            gs = sess.run(g.global_step)
            saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")
