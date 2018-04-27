"""
cbhg.py by TayTech

Contains functions for the CBHG module used in the encoder and after
the decoder. 
"""
import tensorflow as tf
from config import EMBED_SIZE, NUM_HIGHWAY_LAYERS, N_FFT


def cbhg(inputs, lengths, is_training, projections, scope="cbhb", k=16, post=False):
    """
    The CBGB Module used to process the inputs

    :param inputs: list to process
    :param lengths: the length of the inputs
    :param is_training: whether or not the graph is training
    :param scope: used for tf.variable_scope
    :param k: a parameter used in generating the convolutional banks
    :param projections: size of projections
    :param post:
    :return: a rnn with GRU cells for the graph
    """
    with tf.variable_scope(scope):
        # Convolutional banks and max pool
        # The input sequence is first convolved with K sets of 1-D convolutional filters.
        print("Setting up Convolutional banks...")
        banks = conv1d_banks(inputs, k=k, is_training=is_training)
        print("Preparing for a dip with Max pooling...")
        banks = tf.layers.max_pooling1d(banks, pool_size=2, strides=1, padding="same")

        # Conv1D Layers
        print("Hosting your convolutional filters...")
        banks = conv1d(
            banks,
            filters=projections[0],
            kernel_size=3,
            scope="conv1d_1",
            activation_fn=tf.nn.relu,
            is_training=is_training)
        banks = conv1d(
            banks,
            filters=projections[1],
            kernel_size=3,
            scope="conv1d_2",
            activation_fn=None,
            is_training=is_training)

        # Multi-layer highway network
        print("Creating your ultimate Highway network...")
        if post:
            # Extra affine transformation for dimensionality sync
            highway_in = tf.layers.dense(banks, projections[0])  #
        else:
            highway_in = inputs + banks

        for i in range(0, NUM_HIGHWAY_LAYERS):
            highway_in = highwaynet(highway_in, num_units=projections[0], scope=("highway_net" + str(i)))

        print("How would you like a GRU RNN?...")
        # bidirectional GRU RNN to extract sequential features from both
        # forward and backward context
        rnn = cbhg_rnn(inputs, scope=("cbgh_gru_rnn_" + scope))
    return rnn


def cbhg_helper(inputs, lengths, is_training, post=False):
    """
    Helper function for the CBHG module. Specifies different parameters based
    off of the value of post.

    :param inputs: input to process
    :param lengths: length of inputs
    :param is_training: whether or not the graph is training
    :param post: whether to run the post-network cbhg or the encoder cbhg
    :return: the results of the cbhg module
    """
    if post:
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, lengths])

        return tf.layers.dense(cbhg(inputs, None, is_training, scope='post_cbhg', k=8,
                                    projections=[EMBED_SIZE, lengths], post=post), 1+N_FFT//2)
    return cbhg(inputs, lengths, is_training, scope='pre_cbhg', k=16,
                projections=[EMBED_SIZE//2, EMBED_SIZE//2])


def cbhg_rnn(inputs, num_units=EMBED_SIZE//2, scope="cbgh_rnn"):
    """
    Create the RNN with GRUCells for the CBHG module
    Returns the bidirectional rnn

    :param inputs:
    :param num_units:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope):
        # The bidirectional GRU RNN
        gru_rnn, _ = tf.nn.bidirectional_dynamic_rnn(
          tf.contrib.rnn.GRUCell(num_units),
          tf.contrib.rnn.GRUCell(num_units),
          inputs,
          dtype=tf.float32)
    return tf.concat(gru_rnn, 2)


def conv1d(inputs, kernel_size, activation_fn=None, is_training=True, scope="conv1d", filters=None):
    """
    Create the 1-D convolutional layers and normalize.

    Each convolutional layer is batch normalized

    :param inputs: the input features
    :param kernel_size: the size of the kernel of the conv1d
    :param activation_fn: the activation function for the conv1d
    :param is_training: whether or not the graph is training
    :param scope: the scope for variable_scope
    :param filters: filter sizes
    :return: A normalized conv1d layer
    """
    # If no input filters
    if filters is None:
        filters = inputs.get_shape().as_list[-1]
    with tf.variable_scope(scope):
        # Create the conv1d
        conv1d_output = tf.layers.conv1d(inputs,
                                         filters=filters,
                                         kernel_size=kernel_size,
                                         strides=1,
                                         activation=activation_fn,
                                         padding='same')

    # Batch normalization is used for all convolutional layers
    return tf.layers.batch_normalization(conv1d_output, training=is_training)


def conv1d_banks(inputs, k=16, is_training=True, scope="conv1d_banks"):
    """
    This function convolves the input sequence with K sets of 1-D convolutional filters,
    where the k-th set contains C_k filters of width k.
    These filters explicitly model local and contextual information.

    :param inputs: the input sequence
    :param k: the number of sets of convoutional filters
    :param is_training: whether or not the model is training
    :param scope: the scope for tf.variable_scope
    :return: The outputs after the convolutional bank
    """
    with tf.variable_scope(scope):
        # The first convolutional filter
        outputs = conv1d(inputs,  filters=EMBED_SIZE//2, kernel_size=1,
                                    is_training=is_training, scope="conv1d_convbanks1")

        # The next filters until there are k filters
        for k in range(2, k + 1):
            with tf.variable_scope("num_{}".format(k)):
                # convolutional layer
                output = conv1d(inputs, filters=EMBED_SIZE // 2, kernel_size=k,
                                    is_training=is_training, scope="conv1d_convbanks2")
                outputs = tf.concat((outputs, output), -1)
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    return outputs 


def highwaynet(inputs, num_units=None, scope="highwaynet"):
    """
    One layer of the highway network for the CBHG module.

    outputs = H ∗ T + inputs ∗ (1−T)
    Where H is a highway network consisting of multiple blocks
    and T is a transform gate output.

    :param inputs:
    :param num_units:
    :param scope:
    :return: The outputs after one layer of the highway network.
    """
    if not num_units:
        num_units = inputs.get_shape()[-1]
        
    with tf.variable_scope(scope):
        # Values used to construct highway network layer
        H = tf.layers.dense(inputs, units=num_units, activation=tf.nn.relu, name="R2D2")
        T = tf.layers.dense(inputs, units=num_units, activation=tf.nn.sigmoid,
                            bias_initializer=tf.constant_initializer(-1.0), name="D2R2")
        # Highway network layer
        outputs = H*T + inputs*(1.-T)
    return outputs
