"""
eval.py by TayTech

Evaluate the model
"""

import numpy as np
from dataload import input_load
import tensorflow as tf
import os
from train import Model
from utils import create_spectrograms
from config import LOG_DIR, MODEL_NAME
from tqdm import tqdm


def eval_model():
    """
    Evaluate the model
    :return:
    """
    # Load graph
    g = Model(mode="eval")
    print("Evaluation Graph loaded")

    # Load data
    fpaths, text_lengths, texts = input_load(mode="eval")

    # Parse
    text = np.fromstring(texts[0], np.int32)
    fname, mel, mag = create_spectrograms(fpaths[0])

    # Inputs
    text = np.expand_dims(text, 0)
    mels = np.expand_dims(mel, 0)
    mags = np.expand_dims(mag, 0)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # restore the model
        saver = tf.train.import_meta_graph(os.path.join(LOG_DIR, MODEL_NAME))
        saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))
        print("Restored!")

        writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        # Feed Forward
        # mels
        print("Running session...")
        mels_hat = np.zeros((1, mels.shape[1], mels.shape[2]), np.float32)
        for i in tqdm(range(mels.shape[1])):
            _mels_hat = sess.run(g.mel_hat, {g.txt: text, g.mels: mels_hat})
            mels_hat[:, i, :] = _mels_hat[:, i, :]

        # mags
        print("Generating summaries...")
        merged, gs = sess.run(
            [g.merged, g.global_step],
            {g.txt: text, g.mels: mels,
             g.mel_hat: mels_hat, g.mags: mags})

        # summary
        writer.add_summary(merged, global_step=gs)
        writer.close()


if __name__ == '__main__':
    eval_model()
    print("Done")
