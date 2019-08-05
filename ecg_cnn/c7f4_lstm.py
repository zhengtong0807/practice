from keras.layers.core import Reshape
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers import Lambda


def cnn_lstm_model(batch_size):
    filters = 128
    kernel_size = 9
    rate1 = 0.25
    rate2 = 0.5

    n_dim = 15000
    n_split = 300

    NUM_CLASSES = 4

    model = Sequential()

    # cnn_1
    # model.add(Reshape((n_split,1),input_shape=(n_dim,1)))
    model.add(Lambda(lambda x: K.reshape(x,(-1,n_split,1)),input_shape=(n_dim,1)))
    model.add(Conv1D(filters, kernel_size,
                     padding='same',
                     activation='relu',
                     input_shape=(n_split,1)))
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

    model.add(GlobalAveragePooling1D())

    shape_before_lstm=model.output_shape
    print('shape_before_lstm: ',shape_before_lstm)
    # model.add(Reshape((n_dim//n_split,shape_before_lstm[1]*shape_before_lstm[2])))
    model.add(Lambda(lambda x: K.reshape(x,(-1,n_dim//n_split,shape_before_lstm[1]))))
    print('!!!',model.output_shape)
    model.add(Bidirectional(LSTM(72)))
    print('!!!',model.output_shape)
    shape_after_lstm=model.output_shape
    print('shape_after_lstm: ',shape_after_lstm)
    model.add(Dropout(rate2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(rate2))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model