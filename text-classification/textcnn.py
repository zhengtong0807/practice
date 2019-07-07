from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam


def create_model(inputlen, vocabulary, num_class,
                 embedding_dim=128,
                 num_filters=512,
                 filter_sizes=[3, 4, 5],
                 drop_rate=0.5,
                 lr=1e-4):
    inputs = Input(shape=(inputlen,))

    embedding = Embedding(input_dim=vocabulary+1,
                          output_dim=embedding_dim,
                          input_length=inputlen)(inputs)
    reshape = Reshape((inputlen, embedding_dim, 1))(embedding)

    conv_0 = Conv2D(num_filters,
                    kernel_size=(filter_sizes[0], embedding_dim),
                    padding='valid',
                    activation='relu')(reshape)
    conv_1 = Conv2D(num_filters,
                    kernel_size=(filter_sizes[1], embedding_dim),
                    padding='valid',
                    activation='relu')(reshape)
    conv_2 = Conv2D(num_filters,
                    kernel_size=(filter_sizes[2], embedding_dim),
                    padding='valid',
                    activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(inputlen - filter_sizes[0] + 1, 1),
                          strides=(1, 1),
                          padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(inputlen - filter_sizes[1] + 1, 1),
                          strides=(1, 1),
                          padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(inputlen - filter_sizes[2] + 1, 1),
                          strides=(1, 1),
                          padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)(
        [maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop_rate)(flatten)
    output = Dense(units=num_class, activation='softmax')(dropout)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
