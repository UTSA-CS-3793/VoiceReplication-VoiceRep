"""
model.py by TayTech

Contains the class for the main model
"""

from dataload import get_batch
import tensorflow as tf
from networks import encoder, decoder
from utils import spectrogram2wav, learning_rate_decay
from config import N_MELS, REDUCTION_FACTOR, N_FFT, SR
from cbhg import cbhg_helper


class Model:
    """
    Generate the Tensorflow graph
    """
    def __init__(self, mode="train"):
        """
        Initialize the class based off of the given mode
        :param mode: the mode to load the model based on
        """
        print("Loading your model...")

        # Initialize values used in class
        self.mode = mode
        self.global_step = None
        self.mel_loss = None
        self.mel_loss = None
        self.mag_loss = None
        self.learning_rate = None
        self.optimizer = None
        self.merged = None
        self.gradients = None
        self.clipped = None
        self.gvs = None
        self.opt_train = None

        # If is_training
        if mode == "train":
            self.is_training = True
        else:
            self.is_training = False

        print("Loading inputs...")
        # Load inputs
        if self.is_training:
            self.txt, self.mels, self.mags, self.file_names, self.num_batch = get_batch()
        elif mode == "synthesize":
            self.txt = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, N_MELS*REDUCTION_FACTOR))
        else:  # eval
            self.txt = tf.placeholder(tf.int32, shape=(None, None))
            self.mels = tf.placeholder(tf.float32, shape=(None, None, N_MELS*REDUCTION_FACTOR))
            self.mags = tf.placeholder(tf.float32, shape=(None, None, 1 + N_FFT//2))
            self.file_names = tf.placeholder(tf.string, shape=(None,))      

        # decoder inputs
        self.decoder_inputs = tf.concat((tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1)
        self.decoder_inputs = self.decoder_inputs[:, :, -N_MELS:]

        # Networks
        with tf.variable_scope("Networks"):
            print("Loading the encoder...")
            # encoder
            self.memory = encoder(self.txt, is_training=self.is_training)

            print("Loading the decoder...")
            # decoder
            self.mel_hat, self.alignments = decoder(self.decoder_inputs, self.memory, is_training=self.is_training)

            print("Loading the post CBHG module...")
            # CBHG Module
            self.mags_hat = cbhg_helper(self.mel_hat, N_MELS, is_training=self.is_training, post=True)

        print("Audio out")
        # audio
        self.audio_out = tf.py_func(spectrogram2wav, [self.mags_hat[0]], tf.float32)

        # Training and evaluation
        if mode in ("train", "eval"):
            print("Generating Loss...")
            # Loss
            self.loss = self.get_loss()

            print("Getting the optimizer ready...")
            # Training Scheme
            self.optimize()

            print("Setting up your summary...")
            self.summarize()

    def get_loss(self):
        """
        Determine the loss of the outputs
        :return: the loss
        """
        self.mel_loss = tf.reduce_mean(tf.abs(self.mel_hat - self.mels))
        self.mag_loss = tf.reduce_mean(tf.abs(self.mags_hat - self.mags))
        return self.mel_loss + self.mag_loss

    def optimize(self):
        """
        Optimize the learning rate.
        """
        # global step
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        # learning rate decay
        self.learning_rate = learning_rate_decay(global_step=self.global_step)
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Gradient clipping
        gradients, variables = zip(*self.optimizer.compute_gradients(self.loss))
        self.gradients = gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.opt_train = self.optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=self.global_step)

    def summarize(self):
        """
        Summarize the training
        """
        tf.summary.scalar('mode_%s\nmel_loss' % self.mode, self.mel_loss)
        tf.summary.scalar('mag_loss', self.mag_loss)
        tf.summary.scalar('total_loss', self.loss)
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.image('Mel_input', tf.expand_dims(self.mels, -1), max_outputs=1)
        tf.summary.image('Mel_output', tf.expand_dims(self.mel_hat, -1), max_outputs=1)
        tf.summary.image('Mag_input', tf.expand_dims(self.mags, -1), max_outputs=1)
        tf.summary.image('Mag_output', tf.expand_dims(self.mags_hat, -1), max_outputs=1)
        tf.summary.audio('Audio', tf.expand_dims(self.audio_out, 0), SR)
        self.merged = tf.summary.merge_all()
