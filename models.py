from keras.models import Sequential, Model
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Reshape, RNN, Input, Add
from kerasncp.tf import LTCCell
from kerasncp.wirings import NCP
from keras.utils import plot_model

from utils import INPUT_SHAPE


def NVIDIA_model():
    """

    :return:
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 2, activation='relu'))
    model.add(Conv2D(36, 5, 2, activation='relu'))
    model.add(Conv2D(48, 5, 2, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    # model.add(Conv2D(64, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    plot_model(model, show_shapes=True)
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
    plot_model(model, show_shapes=True)
    return model


def stacked_model():
    """
    CNN+NCP stacked model.
    :return: sequential stacked model
    """
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
    rnn_cell = LTCCell(wiring)

    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 2, activation='relu'))
    model.add(Conv2D(36, 5, 2, activation='relu'))
    model.add(Conv2D(48, 5, 2, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    # model.add(Conv2D(64, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Reshape((1, -1)))
    model.add(RNN(rnn_cell, unroll=True))
    model.summary()
    plot_model(model, show_shapes=True)
    return model


# Functional dual NCP model
# As close as possible to the two brain wiring drawings
# gives really high loss value on the comma.ai dataset
def stacked_dual_model():
    """
    Stacked functional CNN and dual NCP model.
    Two NCP wiring settings representing the left and right sides of the human brain.
    The
    :return:
    """
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
    ncp_cell_left = LTCCell(left_wiring)
    ncp_cell_right = LTCCell(right_wiring)

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


# Functional dual NCP model
# Kept the right wiring as given in the Google Colaboratory link below.
# https://colab.research.google.com/drive/1-mZunxqVkfZVBXNPG0kTSKUNQUSdZiBI?usp=sharing#scrollTo=3phJCJCa894e
# The replicated 'left brain' wiring was adjusted taking into account the right side one.
# That is in the brain wiring drawings sent to me by my supervisor showed 4 and 3 inter and
# 6 and 5 command neurons of the right and left sides respectively.
# Therefore, to preserve this ratio, I multiplied the Colab inter neurons by 3/4 and command neurons by 5/6 to get
# the count for the left side wiring replication. The resultant model is provided below.
def stacked_dual_model_v2():
    """
    Stacked functional CNN and dual NCP model v2.
    Two NCP wiring settings representing the left and right sides of the human brain.
    The
    :return:
    """
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

    ncp_cell_left = LTCCell(left_wiring)
    ncp_cell_right = LTCCell(right_wiring)

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
    plot_model(model, show_shapes=True)
    return model
