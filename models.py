"""
Define the CNN benchmark and hybrid model architectures.

This script comprise the code implementations of the developed neural networks.
"""
from keras.models import Sequential, Model
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Reshape, RNN, Input, Add, MaxPool2D
from kerasncp.ltc_cell import LTCCell
from kerasncp.wirings import NCP
from keras.utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt
from constants import INPUT_SHAPE


def plot_wiring(cell):
    """
    Plot the NCP wiring.

    :param cell: NCP cell
    :return: None
    """

    sns.set_style("white")
    plt.figure(figsize=(12, 12))
    legend_handles = cell.draw_graph(layout='spiral', neuron_colors={"command": "tab:cyan"})
    plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.show()


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
    # plot_model(model, show_shapes=True)
    model.summary()
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
        inter_neurons=3,
        command_neurons=5,
        motor_neurons=1,
        sensory_fanout=2,
        inter_fanout=2,
        recurrent_command_synapses=1,
        motor_fanin=2,
    )

    right_wiring = NCP(
        inter_neurons=4,
        command_neurons=6,
        motor_neurons=1,
        sensory_fanout=2,
        inter_fanout=2,
        recurrent_command_synapses=2,
        motor_fanin=2,
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

    # Set the NCP wirings
    left_wiring = NCP(
        inter_neurons=9,
        command_neurons=7,
        motor_neurons=1,
        sensory_fanout=4,
        inter_fanout=4,
        recurrent_command_synapses=4,
        motor_fanin=6,
    )

    right_wiring = NCP(
        inter_neurons=12,
        command_neurons=8,
        motor_neurons=1,
        sensory_fanout=4,
        inter_fanout=4,
        recurrent_command_synapses=4,
        motor_fanin=6,
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

    left_wiring = NCP(
        inter_neurons=12,
        command_neurons=8,
        motor_neurons=1,
        sensory_fanout=4,
        inter_fanout=4,
        recurrent_command_synapses=4,
        motor_fanin=6,
    )

    right_wiring = NCP(
        inter_neurons=5,
        command_neurons=3,
        motor_neurons=1,
        sensory_fanout=4,
        inter_fanout=2,
        recurrent_command_synapses=2,
        motor_fanin=2,
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

    left_wiring = NCP(
        inter_neurons=12,
        command_neurons=8,
        motor_neurons=1,
        sensory_fanout=4,
        inter_fanout=4,
        recurrent_command_synapses=4,
        motor_fanin=6,
    )

    right_wiring = NCP(
        inter_neurons=5,
        command_neurons=3,
        motor_neurons=1,
        sensory_fanout=4,
        inter_fanout=2,
        recurrent_command_synapses=2,
        motor_fanin=2,
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
    dropout_lyr = Dropout(0.5)(conv_3)
    flatten_lyr = Flatten()(dropout_lyr)
    dense_1 = Dense(100, activation='relu')(flatten_lyr)
    reshape_lyr = Reshape((1, -1))(dense_1)
    ncp_left = RNN(ncp_cell_left, unroll=True)(reshape_lyr)
    ncp_right = RNN(ncp_cell_right, unroll=True)(reshape_lyr)
    ncps_added = Add()([ncp_left, ncp_right])
    output = Lambda(lambda x: x * 0.5)(ncps_added)
    model = Model(inputs=input_lyr, outputs=output)
    # model.summary()
    # plot_model(model, show_shapes=True)
    return model
