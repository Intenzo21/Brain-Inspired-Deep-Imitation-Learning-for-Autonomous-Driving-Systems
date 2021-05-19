"""
Define the CNN benchmark and hybrid model architectures.

This script comprise the code implementations of the developed neural networks.
"""
from keras.models import Sequential, Model
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Reshape, RNN, Input, Add, MaxPool2D
from kerasncp.tf import LTCCell
from kerasncp.wirings import NCP
from keras.utils import plot_model

from constants import INPUT_SHAPE


def cnn_model():
    """
    Define the Keras sequential Nvidia CNN benchmark architecture.

    :return: the CNN reference model
    """

    # Build the sequential CNN model
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 2, activation='relu'))
    model.add(Conv2D(36, 5, 2, activation='relu'))
    model.add(Conv2D(48, 5, 2, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    # plot_model(model, show_shapes=True)
    return model


def other_model():
    """
    Random (personally developed) CNN model for testing.

    :return: sequential CNN model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.,
                     input_shape=INPUT_SHAPE,
                     output_shape=INPUT_SHAPE))
    model.add(Conv2D(8, 3, 2, padding="same", activation='relu'))
    model.add(Conv2D(16, 3, 2, padding="same", activation='relu'))
    model.add(Conv2D(32, 3, 2, padding="same", activation='relu'))
    model.add(Conv2D(64, 3, 2, padding="same", activation='relu'))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(.5))
    model.add(Dense(1, activation='relu'))
    model.summary()
    # plot_model(model, show_shapes=True)
    return model


def cnn_ncp_model():
    """
    Define the Keras sequential CNN-NCP hybrid architecture.

    A CNN feature extractor is stacked with an NCP RNN temporal modelling structure.

    :return: the CNN-NCP model
    """

    # Set the NCP wiring
    wiring = NCP(
        inter_neurons=12,  # Number of inter neurons
        command_neurons=8,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming synapses has each motor neuron
    )

    # Create the the NCP cell based on the LTC neuron.
    ncp_cell = LTCCell(wiring)

    # Build the sequential hybrid model
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 2, activation='relu'))
    model.add(Conv2D(36, 5, 2, activation='relu'))
    model.add(Conv2D(48, 5, 2, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Reshape((1, -1)))
    model.add(RNN(ncp_cell, unroll=True))
    model.summary()
    # plot_model(model, show_shapes=True)
    return model


def cnn_dncp_model():
    """
    Define the blended functional CNN-DNCP v2 model.

    Two NCP wiring settings are adopted representing the left and right sides of the human brain.
    This model is, therefore, called DNCP which is short for Dual NCP.

    :return: the brain-inspired CNN-DNCP model
    """

    # Set the NCP wiring configurations
    left_wiring = NCP(
        inter_neurons=3,  # Number of inter neurons
        command_neurons=5,  # Number of command neurons
        motor_neurons=1,  # (1 output) Number of motor neurons
        sensory_fanout=2,  # How many outgoing synapses has each sensory neuron
        inter_fanout=2,  # (average=15/8) How many outgoing synapses has each inter neuron
        recurrent_command_synapses=1,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=2,  # (12/5) How many incoming synapses has each motor neuron
    )

    right_wiring = NCP(
        inter_neurons=4,  # Number of inter neurons
        command_neurons=6,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=2,  # (18/8) How many outgoing synapses has each sensory neuron
        inter_fanout=2,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=2,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=2,  # How many incoming synapses has each motor neuron
    )

    # Create the left and right brain-inspired NCP cells.
    # These are based on the LTC neuron.
    ncp_cell_left = LTCCell(left_wiring)
    ncp_cell_right = LTCCell(right_wiring)

    # Build the functional brain-inspired hybrid model
    input_lyr = Input(shape=INPUT_SHAPE)
    lambda_lyr = Lambda(lambda x: x / 127.5 - 1.0)(input_lyr)
    conv_1 = Conv2D(24, 5, 2, activation='relu')(lambda_lyr)
    conv_2 = Conv2D(36, 5, 2, activation='relu')(conv_1)
    conv_3 = Conv2D(48, 5, 2, activation='relu')(conv_2)
    conv_4 = Conv2D(64, 3, activation='relu')(conv_3)
    conv_5 = Conv2D(64, 3, activation='relu')(conv_4)
    dropout_lyr = Dropout(0.5)(conv_5)
    flatten_lyr = Flatten()(dropout_lyr)
    dense_1 = Dense(100, activation='relu')(flatten_lyr)
    reshape_lyr = Reshape((1, -1))(dense_1)
    ncp_left = RNN(ncp_cell_left, unroll=True)(reshape_lyr)
    ncp_right = RNN(ncp_cell_right, unroll=True)(reshape_lyr)
    ncps_added = Add()([ncp_left, ncp_right])
    output = Lambda(lambda x: x * 0.5)(ncps_added)
    model = Model(inputs=input_lyr, outputs=output)
    model.summary()
    # plot_model(model, show_shapes=True)
    return model


def cnn_dncp_v2_model():
    """
    Define the blended functional version 2 of the CNN-DNCP model.

    This model is built the same way as the original CNN-DNCP but with
    modified NCP wiring configurations.

    :return: the CNN-DNCP v2 model
    """

    # Set the NCP wiring configurations
    left_wiring = NCP(
        inter_neurons=9,  # Number of inter neurons
        command_neurons=7,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming synapses has each motor neuron
    )

    right_wiring = NCP(
        inter_neurons=12,  # Number of inter neurons
        command_neurons=8,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming synapses has each motor neuron
    )

    # Create the left and right brain-inspired NCP cells.
    # These are based on the LTC neuron.
    ncp_cell_left = LTCCell(left_wiring)
    ncp_cell_right = LTCCell(right_wiring)

    # Build the functional brain-inspired hybrid model
    input_lyr = Input(shape=INPUT_SHAPE)
    lambda_lyr = Lambda(lambda x: x / 127.5 - 1.0)(input_lyr)
    conv_1 = Conv2D(24, 5, 2, activation='relu')(lambda_lyr)
    conv_2 = Conv2D(36, 5, 2, activation='relu')(conv_1)
    conv_3 = Conv2D(48, 5, 2, activation='relu')(conv_2)
    conv_4 = Conv2D(64, 3, activation='relu')(conv_3)
    conv_5 = Conv2D(64, 3, activation='relu')(conv_4)
    dropout_lyr = Dropout(0.5)(conv_5)
    flatten_lyr = Flatten()(dropout_lyr)
    dense_1 = Dense(100, activation='relu')(flatten_lyr)
    reshape_lyr = Reshape((1, -1))(dense_1)
    ncp_left = RNN(ncp_cell_left, unroll=True)(reshape_lyr)
    ncp_right = RNN(ncp_cell_right, unroll=True)(reshape_lyr)
    ncps_added = Add()([ncp_left, ncp_right])
    output = Lambda(lambda x: x * 0.5)(ncps_added)
    model = Model(inputs=input_lyr, outputs=output)
    model.summary()
    return model


def cnn_dncp_v3_model():
    """
    Define the blended functional version 3 of the CNN-DNCP model.

    This model is built the same way as the original CNN-DNCP but with
    modified NCP wiring configurations.

    :return: the CNN-DNCP v3 model
    """

    # Set the NCP wiring configurations
    left_wiring = NCP(
        inter_neurons=12,  # Number of inter neurons
        command_neurons=8,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming synapses has each motor neuron
    )

    right_wiring = NCP(
        inter_neurons=5,  # Number of inter neurons
        command_neurons=3,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=2,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=2,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=2,  # How many incoming synapses has each motor neuron
    )

    # Create the left and right brain-inspired NCP cells.
    # These are based on the LTC neuron.
    ncp_cell_left = LTCCell(left_wiring)
    ncp_cell_right = LTCCell(right_wiring)

    # Build the functional brain-inspired hybrid model
    input_lyr = Input(shape=INPUT_SHAPE)
    lambda_lyr = Lambda(lambda x: x / 127.5 - 1.0)(input_lyr)
    conv_1 = Conv2D(24, 5, 2, activation='relu')(lambda_lyr)
    conv_2 = Conv2D(36, 5, 2, activation='relu')(conv_1)
    conv_3 = Conv2D(48, 5, 2, activation='relu')(conv_2)
    conv_4 = Conv2D(64, 3, activation='relu')(conv_3)
    conv_5 = Conv2D(64, 3, activation='relu')(conv_4)
    dropout_lyr = Dropout(0.5)(conv_5)
    flatten_lyr = Flatten()(dropout_lyr)
    dense_1 = Dense(100, activation='relu')(flatten_lyr)
    reshape_lyr = Reshape((1, -1))(dense_1)
    ncp_left = RNN(ncp_cell_left, unroll=True)(reshape_lyr)
    ncp_right = RNN(ncp_cell_right, unroll=True)(reshape_lyr)
    ncps_added = Add()([ncp_left, ncp_right])
    output = Lambda(lambda x: x * 0.5)(ncps_added)
    model = Model(inputs=input_lyr, outputs=output)
    model.summary()
    return model


def cnn_dncp_v4_model():
    """
    Define the blended functional version 4 of the CNN-DNCP model.

    In this model, two of the CNN-DNCP CNN layers are removed to make way
    for the two MaxPool2D layers to reduce its complexity.
    The same NCP wiring settings are used as in the CNN-DNCP v4 model.

    :return: the CNN-DNCP v4 model
    """

    # Set the NCP wiring configurations
    left_wiring = NCP(
        inter_neurons=12,  # Number of inter neurons
        command_neurons=8,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=4,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=4,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=6,  # How many incoming synapses has each motor neuron
    )

    right_wiring = NCP(
        inter_neurons=5,  # Number of inter neurons
        command_neurons=3,  # Number of command neurons
        motor_neurons=1,  # Number of motor neurons
        sensory_fanout=4,  # How many outgoing synapses has each sensory neuron
        inter_fanout=2,  # How many outgoing synapses has each inter neuron
        recurrent_command_synapses=2,  # Now many recurrent synapses are in the
        # command neuron layer
        motor_fanin=2,  # How many incoming synapses has each motor neuron
    )

    # Create the left and right brain-inspired NCP cells.
    # These are based on the LTC neuron.
    ncp_cell_left = LTCCell(left_wiring)
    ncp_cell_right = LTCCell(right_wiring)

    # Build the functional brain-inspired hybrid model
    input_lyr = Input(shape=INPUT_SHAPE)
    lambda_lyr = Lambda(lambda x: x / 127.5 - 1.0)(input_lyr)
    conv_1 = Conv2D(24, 5, 2, activation='relu')(lambda_lyr)
    max_1 = MaxPool2D()(conv_1)
    conv_2 = Conv2D(48, 5, 2, activation='relu')(max_1)
    max_2 = MaxPool2D()(conv_2)
    conv_3 = Conv2D(64, 3, activation='relu')(max_2)
    max_3 = MaxPool2D()(conv_3)
    dropout_lyr = Dropout(0.5)(max_3)
    flatten_lyr = Flatten()(dropout_lyr)
    dense_1 = Dense(100, activation='relu')(flatten_lyr)
    reshape_lyr = Reshape((1, -1))(dense_1)
    ncp_left = RNN(ncp_cell_left, unroll=True)(reshape_lyr)
    ncp_right = RNN(ncp_cell_right, unroll=True)(reshape_lyr)
    ncps_added = Add()([ncp_left, ncp_right])
    output = Lambda(lambda x: x * 0.5)(ncps_added)
    model = Model(inputs=input_lyr, outputs=output)
    model.summary()
    # plot_model(model, show_shapes=True)
    return model
