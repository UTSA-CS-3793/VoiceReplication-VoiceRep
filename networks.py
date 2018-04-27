"""
networks.py by TayTech

Includes the encoder and decoder RNNs
"""
import tensorflow as tf 
from config import EMBED_SIZE, VOCAB_SIZE, DROPOUT_RATE, N_MELS, REDUCTION_FACTOR
from cbhg import cbhg_helper
from utils import attention


def encoder(inputs, is_training=True, scope="encoder"):
    """
    Encoder for the input sequence.
    Embeds the character sequence -> Runs through the pre-net -> CBHG Module

    :param inputs: inputs for the model
    :param is_training: whether or not the model is training
    :param scope:
    :return: the results
    """
    # Get encoder/decoder inputs
    print("Getting encoder inputs...")
    encoder_inputs = embed(inputs, VOCAB_SIZE, EMBED_SIZE)
    # Networks
    with tf.variable_scope(scope):
        # Encoder pre-net
        print("Generating the Pre-Net...")
        pre_out = pre_net(encoder_inputs, is_training=is_training)
        # Run CBHG
        print("Loading the CBHG...")
        cbhg_net = cbhg_helper(inputs=pre_out, lengths=EMBED_SIZE//2, is_training=is_training)
    return cbhg_net


def embed(inputs, vocab_size, num_units, scope="embedding"):
    """
    Embeds character sequence into a continuous vector

    :param inputs: inputs for embedding
    :param vocab_size: size of the vocabulary
    :param num_units:
    :param scope:
    :return: the embedded lookup
    """
    with tf.variable_scope(scope):
        # Create a look-up table
        lookup_table = tf.get_variable('lookup_table', 
                                       dtype=tf.float32, 
                                       shape=[vocab_size, num_units],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        
        # Concatenate the tensors along one-dimension
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)

    return tf.nn.embedding_lookup(lookup_table, inputs)


def pre_net(inputs, is_training=True, num_hidden_units=None):
    """
    Apply a set of non-linear transformations, collectively called a "pre-net",
    to each embedding.

    Used a bottleneck layer with dropout, which helps convergence and improves generalization.

    :param inputs: inputs for the pre-net layers
    :param is_training: whether or not the model is training
    :param num_hidden_units: the number of hidden units
    :return: the outputs from the layers
    """
    if num_hidden_units is None:
        num_hidden_units = [EMBED_SIZE, EMBED_SIZE // 2]
    
    # Apply the series of transformations
    outputs = inputs
    for i in range(len(num_hidden_units)):
        outputs = tf.layers.dense(outputs, units=num_hidden_units[i], activation=tf.nn.relu, name=("dense" + str(i)))
        outputs = tf.layers.dropout(outputs, rate=DROPOUT_RATE, training=is_training, name=("dropout" + str(i)))
    return outputs


def decoder(inputs, memory, is_training=True, scope="decoder"):
    """
    A content-based tanh attention decoder using a stack of GRUs with vertical
    residual connections.

    Takes the output from the encoder, runs it though a prenet,
    then processes with attention. After finishing the attention,
    generates the decoder RNN.

    Although the decoder could directly target the raw spectogram, this would
    be a highly redundant representation for the purpose of learning alignment
    between speech signal and text. Thus the target is an 80-band mel-scale
    spectogram, though fewer bands or more concise targets such as
    cepstrum could be used.

    :param inputs:
    :param memory:
    :param is_training:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        # prenet
        inputs = pre_net(inputs, is_training=is_training)

        # With Attention
        outputs, state = attention(inputs, memory, num_units=EMBED_SIZE)

        # Transpose
        alignments = tf.transpose(state.alignment_history.stack(), [1, 2, 0])

        # Decoder RNNs - 2-Layer Residual GRU (256 cells)
        outputs += decoder_rnn(outputs, scope="decoder_rnn1")
        outputs += decoder_rnn(outputs, scope="decoder_rnn2")
        
        # An 80-band mel-scale spectogram is the target
        mel_hats = tf.layers.dense(outputs, N_MELS*REDUCTION_FACTOR)
    return mel_hats, alignments


def decoder_rnn(inputs, scope="decoder_rnn"):
    """
    An RNN with GRU cells used in the decoder

    :param inputs:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        rnn, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(EMBED_SIZE), inputs, dtype=tf.float32)

    return rnn
