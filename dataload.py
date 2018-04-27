# -*- coding: utf-8 -*-
"""
dataload.py by TayTech

Load the data for the model.

Is used in eval.py, synthesize.py, and model.py
"""
import numpy as np
import tensorflow as tf
from utils import create_vocab, normalize_text, create_spectrograms
import codecs
import os
from config import N_MELS, REDUCTION_FACTOR, N_FFT, BATCH_SIZE, DATA_PATH, DEVICE, ENCODING, NUM_EPOCHS, TEST_DATA


def input_load(mode="train"):
    """
    Load the input text and the corresponding feature labels

    :param mode: whether to gather data for training and evaluation or for synthesis
    :return: the text labels, the text lengths, and the audio file paths
    """
    # creates vocab conversion dictionaries
    char2idx, _ = create_vocab()
    fpaths, text_lengths, texts = [], [], []

    # the path to the dataset
    base_path = os.path.join(DATA_PATH, 'wavs')
    # the path to the text
    transcript = os.path.join(DATA_PATH, 'metadata.csv')

    # training or evaluation
    if mode in ("train", "eval"):
        # Each epoch
        for _ in range(NUM_EPOCHS):
            # open the text file
            lines = codecs.open(transcript, 'r', ENCODING).readlines()
            for line in lines:
                fname, _, text = line.strip().split("|")

                # get the wav file paths
                fpath = os.path.join(base_path, fname + ".wav")
                fpaths.append(fpath)

                # clean and normalize the text
                text = normalize_text(text) + "$"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())
        return fpaths, text_lengths, texts
    else:           # synthesis

        # Parse
        lines = codecs.open(TEST_DATA, 'r', 'utf-8').readlines()[1:]

        # Normalize text: $ is EOS
        sents = [normalize_text(line.split(" ", 1)[-1]).strip() + "$" for line in lines]
        lengths = [len(sent) for sent in sents]
        maxlen = sorted(lengths, reverse=True)[0]

        # Pad the text
        texts = np.zeros((len(sents), maxlen), np.int32)
        for i, sent in enumerate(sents):
            texts[i, :len(sent)] = [char2idx[char] for char in sent]
        # return just the text, no lengths or paths needed
        return texts


def get_batch():
    """
    Get the batch inputs for the model

    :return: the inputs
    """
    with tf.device(DEVICE):  # You should set the DEVICE in the config file

        # load the data
        fpaths, text_lengths, texts = input_load()
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # determine number of batches
        num_batch = len(fpaths) // BATCH_SIZE

        # Tensors
        fpaths = tf.convert_to_tensor(fpaths)
        text_lengths = tf.convert_to_tensor(text_lengths)
        texts = tf.convert_to_tensor(texts)

        # form queue's from lists
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # For creating spectrograms
        fname, mel, mag = tf.py_func(create_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        # Set shape
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, N_MELS*REDUCTION_FACTOR))
        mag.set_shape((None, N_FFT//2+1))

        # Get buckets
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=BATCH_SIZE,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=16,
                                            capacity=BATCH_SIZE * 4,
                                            dynamic_pad=True)

        return texts, mels, mags, fnames, num_batch
