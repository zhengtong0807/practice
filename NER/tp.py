"""
训练、预测
"""


import config
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model


class NER():
    def __init__(self):
        self.total_sample = []
        self.total_label = []

    def load_data(self):
        maxlen = 0

        with open('data/BosonNLP_NER_6C_process.txt', encoding='utf8') as f:
            for line in f.readlines():
                word_list = line.strip().split()
                one_sample, one_label = zip(
                    *[word.rsplit('/', 1) for word in word_list])
                one_sample_len = len(one_sample)
                if one_sample_len > maxlen:
                    maxlen = one_sample_len
                one_sample = ' '.join(one_sample)
                one_label = [config.classes[label] for label in one_label]
                self.total_sample.append(one_sample)
                self.total_label.append(one_label)

        tok = Tokenizer()
        tok.fit_on_texts(self.total_sample)
        self.vocabulary = len(tok.word_index) + 1
        self.total_sample = tok.texts_to_sequences(self.total_sample)

        self.total_sample = np.array(pad_sequences(
            self.total_sample, maxlen=maxlen, padding='post', truncating='post'))
        self.total_label = np.array(pad_sequences(
            self.total_label, maxlen=maxlen, padding='post', truncating='post'))[:, :, None]

        print('total_sample shape:', self.total_sample.shape)
        print('total_label shape:', self.total_label.shape)

        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.total_sample, self.total_label, test_size=config.proportion['test'], random_state=666)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_train, y_train, test_size=config.proportion['val'], random_state=666)

        print('X_train shape:', self.X_train.shape)
        print('y_train shape:', self.y_train.shape)
        print('X_val shape:', self.X_val.shape)
        print('y_val shape:', self.y_val.shape)
        print('X_test shape:', self.X_test.shape)
        print('y_test shape:', self.y_test.shape)

        del self.total_sample
        del self.total_label

    def build_model(self):
        model = Sequential()

        model.add(Embedding(self.vocabulary, 100, mask_zero=True))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(CRF(len(config.classes), sparse_target=True))
        model.summary()

        opt = Adam(lr=config.hyperparameter['learning_rate'])
        model.compile(opt, loss=crf_loss, metrics=[crf_viterbi_accuracy])

        self.model = model

    def train(self):
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = '{epoch:03d}_{val_crf_viterbi_accuracy:.4f}.h5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        tensorboard = TensorBoard()
        checkpoint = ModelCheckpoint(os.path.join(save_dir, model_name),
                                     monitor='val_crf_viterbi_accuracy',
                                     save_best_only=True)
        lr_reduce = ReduceLROnPlateau(
            monitor='val_crf_viterbi_accuracy', factor=0.2, patience=10)

        self.model.fit(self.X_train, self.y_train,
                       batch_size=config.hyperparameter['batch_size'],
                       epochs=config.hyperparameter['epochs'],
                       callbacks=[tensorboard, checkpoint, lr_reduce],
                       validation_data=[self.X_val, self.y_val])

    def evaluate(self):
        best_model_name = sorted(os.listdir('saved_models')).pop()
        self.best_model = load_model(os.path.join('saved_models', best_model_name),
                                     custom_objects={'CRF': CRF,
                                                     'crf_loss': crf_loss,
                                                     'crf_viterbi_accuracy': crf_viterbi_accuracy})
        scores = self.best_model.evaluate(self.X_test, self.y_test)
        print('test loss:', scores[0])
        print('test accuracy:', scores[1])

    def predict_case(self):
        pass


if __name__ == "__main__":
    ner = NER()

    ner.load_data()
    ner.build_model()
    ner.train()
    ner.evaluate()
    ner.predict_case()
