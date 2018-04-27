"""
train.py by TayTech

Train the model
"""

from model import Model
import tensorflow as tf
from utils import plot_alignments
from config import CHECK_VALS, LOG_DIR, SR, MODEL_NAME
import librosa
import os
from tqdm import tqdm
import time


def train():
    """
    Load the model from a checkpoint and train it.

    Saves the model periodically based on the value of CHECK_VALS
    """
    # Stats
    time_count = 0
    time_sum = 0
    loss_count = 0
    loss_sum = 0

    # Paths
    check_path = os.path.join(LOG_DIR, 'model')
    check_path2 = os.path.join(LOG_DIR, MODEL_NAME)

    g = Model()
    print("Graph for training loaded.")

    # Session
    with tf.Session() as sess:

        # run and initialize
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess=sess, coord=coord)

        # Load the model
        saver = tf.train.Saver()
        if os.path.isfile(check_path2):
            print("LOADED")
            saver = tf.train.import_meta_graph(check_path2)
            saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

        # Run the session
        for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
            start_time = time.time()

            # training
            g_step, g_loss, g_opt = sess.run([g.global_step, g.loss, g.opt_train])

            # Generate stats
            time_count += 1
            loss_count += 1
            time_sum += time.time() - start_time
            loss_sum += g_loss

            # Message
            message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % \
                      (g_step, time_sum/time_count, g_loss, loss_sum/loss_count)
            print(message)

            # Save
            if g_step % CHECK_VALS == 0:

                print("Saving checkpoint to %s at step %d" % (check_path, g_step))
                saver.save(sess, check_path, global_step=g_step)

                # Saving the audio and alignment
                print('Saving audio and alignment...')
                audio_out, alignments = sess.run([g.audio_out, g.alignments[0]])

                # The wav file
                librosa.output.write_wav(os.path.join(LOG_DIR, 'step-%d-audio.wav' % g_step), audio_out, SR)

                # plot alignments
                plot_alignments(alignments, global_step=g_step)


if __name__ == '__main__':
    train()
    print("Done")
