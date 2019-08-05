"""neural networks"""

import sys

import keras
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense
from keras.layers import GlobalAveragePooling1D, BatchNormalization, Activation
from keras.regularizers import l2
from keras.models import Model

NUM_CLASSES = 5


def c7f4():
    """7 conv1d and 4 dense"""
    filters = 128
    kernel_size = 25
    rate1 = 0.25
    rate2 = 0.5

    model = Sequential()

    # cnn_1
    model.add(Conv1D(filters, kernel_size,
                     padding='same',
                     activation='relu',
                     input_shape=(None, 12)))
    model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(rate1))

    # cnn_2
    model.add(Conv1D(filters, kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(rate1))

    # cnn_3
    model.add(Conv1D(filters, kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(rate1))

    # cnn_4
    model.add(Conv1D(filters, kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(rate1))

    # cnn_5
    model.add(Conv1D(filters, kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(rate1))

    # cnn_6
    model.add(Conv1D(filters, kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(rate1))

    # cnn_7
    model.add(Conv1D(filters, kernel_size,
                     padding='same',
                     activation='relu'))
    model.add(MaxPooling1D(padding='same'))
    model.add(Dropout(rate1))

    # Global Average Pooling
    model.add(GlobalAveragePooling1D())

    # fc_1
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate2))

    # fc_2
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate2))

    # fc_3
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(rate2))

    # fc_4
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model, sys._getframe().f_code.co_name


class ResNet(object):
    """build resnet (proposed res unit)

       e.g. :
           non bottleneck :
           18-layer : stacks = [2, 2, 2, 2]
           34-layer : stacks = [3, 4, 6, 3]

           bottleneck :
           50-layer : stacks = [3, 4, 6, 3]
           101-layer : stacks = [3, 4, 23, 3]

       # Arguments
           stacks (list): how many residual unit per stack
           input_shape (tuple): shape of input
           num_classes (int): number of classes

       # Returns
           model (Model): Keras model instance
    """

    def __init__(self, stacks, input_shape, num_classes):
        self.stacks = stacks
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _blocks_non_bottleneck(self, idx, stack, channels_out, x):
        for i in range(stack):
            strides = 1
            if idx != 0 and i == 0:
                strides = 2

            y = BatchNormalization()(x)
            y = Activation('relu')(y)
            y = Conv1D(channels_out, 3,
                       strides=strides,
                       padding='same',
                       kernel_regularizer=l2(1e-4))(y)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv1D(channels_out, 3,
                       padding='same',
                       kernel_regularizer=l2(1e-4))(y)

            if idx != 0 and i == 0:
                x = Conv1D(channels_out, 1,
                           strides=strides,
                           padding='same',
                           kernel_regularizer=l2(1e-4))(x)

            x = keras.layers.add([x, y])

        return x

    def _blocks_bottleneck(self):
        pass

    def build(self):
        inputs = Input(shape=self.input_shape)

        x = Conv1D(64, 7,
                   strides=2,
                   padding='same',
                   activation='relu',
                   kernel_regularizer=l2(1e-4))(inputs)
        x = MaxPooling1D(pool_size=3,
                         strides=2,
                         padding='same')(x)

        channels_out = 64
        for idx, stack in enumerate(self.stacks):
            x = self._blocks_non_bottleneck(idx, stack, channels_out, x)
            channels_out *= 2
        x = GlobalAveragePooling1D()(x)

        outputs = Dense(self.num_classes,
                        activation='softmax',
                        kernel_regularizer=l2(1e-4))(x)

        model = Model(inputs=inputs, outputs=outputs)

        return model
