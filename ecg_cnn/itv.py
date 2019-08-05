"""inference & train & validate"""

import os
import time
import math
import copy
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, recall_score
from sklearn import utils
import tensorflow as tf

import keras
import keras.backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from dp import data_process
from nets import c7f4

sns.set_style('darkgrid')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))

RHYTHM_LIST = ['N', 'A', 'F', 'S', 'Fz']

INIT_LR = 5e-4
BATCH_SIZE = 64
EPOCHS = 100


def generate_batch(files, labels, steps, shuffle=True):
    while True:
        if shuffle:
            files, labels = utils.shuffle(files, labels)

        for i in range(steps):
            slice_files = files[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            slice_labels = labels[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            slice_x, slice_y = data_process(slice_files, slice_labels)

            yield slice_x, slice_y


class AddTensorBoard(TensorBoard):
    """add val_f1 to tensorboard"""

    def __init__(self, val_files, val_labels, val_steps):
        self.val_files = val_files
        self.val_labels = val_labels
        self.val_steps = val_steps
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        val_pred = np.argmax(self.model.predict_generator(generate_batch(self.val_files,
                                                                         self.val_labels,
                                                                         self.val_steps,
                                                                         shuffle=False), 
                                                                         steps=self.val_steps), 
                                                                         axis=1)
        val_f1 = f1_score(self.val_labels, val_pred, average='macro') # unweighted mean

        val_true = copy.copy(self.val_labels)
        val_true[np.where(val_true != 0)[0]] = 1  # np.where(val_true != 0) return tuple
        val_pred[np.where(val_pred != 0)[0]] = 1
        val_fnr = 1 - recall_score(val_true, val_pred, average=None)[1]

        logs.update({'val_f1': val_f1, 'val_fnr': val_fnr})
        super().on_epoch_end(epoch, logs)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues,
                          pic_name='cm.jpg'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(9, 9))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig('pics/' + pic_name)


def main():
    train = pd.read_csv('../csv_files/train.csv')
    val = pd.read_csv('../csv_files/val.csv')
    test = pd.read_csv('../csv_files/test.csv')

    train_files = train['file_name'].values
    train_labels = train['label'].values
    train_steps = train.shape[0] // BATCH_SIZE

    val_files = val['file_name'].values
    val_labels = val['label'].values
    val_steps = math.ceil(val.shape[0] / BATCH_SIZE)

    test_files = test['file_name'].values
    test_labels = test['label'].values
    test_steps = math.ceil(test.shape[0] / BATCH_SIZE)

    # x_val, y_val = data_process(val_files, val_labels)  # Reading too many files at once, then the slower the reading

    del train
    del val
    del test

    print('\nStart training neural network and validate ...')
    itv_start_time = time.time()

    model, net_name = c7f4()
    model.summary()

    opt = keras.optimizers.Adam(lr=INIT_LR)
    model.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = '%s_{epoch:03d}_{val_acc:.4f}_{val_fnr:.4f}.h5' % net_name
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    tensorboard = AddTensorBoard(val_files, val_labels, val_steps)
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_fnr',
                                 save_best_only=True,
                                 mode='min')
    lr_reduce = ReduceLROnPlateau(monitor='val_fnr', factor=0.2, patience=20, mode='min', min_lr=0)
    callbacks = [tensorboard, checkpoint, lr_reduce]  # list order is important !!!

    history = model.fit_generator(generate_batch(train_files, train_labels, train_steps),
                                  steps_per_epoch=train_steps,
                                  epochs=EPOCHS,
                                  verbose=2,
                                  callbacks=callbacks,
                                  validation_data=generate_batch(val_files, val_labels, val_steps, shuffle=False),
                                  validation_steps=val_steps)

    print("\nOptimization Finished! Time used: %.2fh\n" % ((time.time() - itv_start_time) / 3600))
    print('final lr:', history.history['lr'][-1])

    plt.figure(figsize=(16, 9))
    plt.subplot(131)
    plt.plot(history.history['lr'])
    plt.xlabel('epochs')
    plt.title('learning rate')
    plt.subplot(132)
    plt.plot(history.history['acc'], 'r')
    plt.plot(history.history['val_acc'], 'b')
    plt.plot(history.history['val_f1'], 'g')
    plt.legend(['train acc', 'val acc', 'val f1'], loc='best')
    plt.xlabel('epochs')
    plt.title('acc & val f1')
    plt.subplot(133)
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'b')
    plt.plot(history.history['val_fnr'], 'g')
    plt.legend(['train loss', 'val loss', 'val fnr'], loc='best')
    plt.xlabel('epochs')
    plt.title('loss & val fnr')
    plt.savefig('pics/tv_curve.jpg')

    del model
    best_model = sorted(os.listdir('saved_models')).pop()
    model = load_model(os.path.join('saved_models', best_model))

    # 5 classes
    val_true = val_labels
    val_pred_matrix = model.predict_generator(generate_batch(val_files, val_labels, val_steps, shuffle=False), 
                                              steps=val_steps)
    val_pred = np.argmax(val_pred_matrix, axis=1)
    with open('results/val_results.txt', mode='w') as f:
        f.write('5 CLASSES\n')
        f.write('val acc: %.5f\n' % accuracy_score(val_true, val_pred))
        f.write('mean f1: %.5f\n' % f1_score(val_true, val_pred, average='macro'))
        f.write('\n')
        f.write(classification_report(val_true, val_pred, target_names=RHYTHM_LIST, digits=5))
        f.write('\n\n')
    cnf_matrix = confusion_matrix(val_true, val_pred)
    plot_confusion_matrix(cnf_matrix, RHYTHM_LIST, pic_name='cm5_val.jpg')
    # 2 classes
    val_true[np.where(val_true != 0)[0]] = 1
    val_pred[np.where(val_pred != 0)[0]] = 1
    val_fnr = 1 - recall_score(val_true, val_pred, average=None)[1]
    with open('results/val_results.txt', mode='a') as f:
        f.write('2 CLASSES\n')
        f.write('val acc: %.5f\n' % accuracy_score(val_true, val_pred))
        f.write('mean f1: %.5f\n' % f1_score(val_true, val_pred, average='macro'))
        f.write('\n')
        f.write(classification_report(val_true, val_pred, target_names=['N', 'abN'], digits=5))
        f.write('\nval  false negative rate: %.5f\n' % val_fnr)
        f.write('\n\n')
    cnf_matrix = confusion_matrix(val_true, val_pred)
    plot_confusion_matrix(cnf_matrix, ['N', 'abN'], pic_name='cm2_val.jpg')

    # 5 classes
    test_true = test_labels
    test_pred_matrix = model.predict_generator(generate_batch(test_files, test_labels, test_steps, shuffle=False), steps=test_steps)
    test_pred = np.argmax(test_pred_matrix, axis=1)
    with open('results/test_results.txt', mode='w') as f:
        f.write('5 CLASSES\n')
        f.write('test acc: %.5f\n' % accuracy_score(test_true, test_pred))
        f.write('mean  f1: %.5f\n' % f1_score(test_true, test_pred, average='macro'))
        f.write('\n')
        f.write(classification_report(test_true, test_pred, target_names=RHYTHM_LIST, digits=5))
        f.write('\n\n')
    cnf_matrix = confusion_matrix(test_true, test_pred)
    plot_confusion_matrix(cnf_matrix, RHYTHM_LIST, pic_name='cm5_test.jpg')
    # 2 classes
    test_true[np.where(test_true != 0)[0]] = 1
    test_pred[np.where(test_pred != 0)[0]] = 1
    test_fnr = 1 - recall_score(test_true, test_pred, average=None)[1]
    with open('results/test_results.txt', mode='a') as f:
        f.write('2 CLASSES\n')
        f.write('test acc: %.5f\n' % accuracy_score(test_true, test_pred))
        f.write('mean  f1: %.5f\n' % f1_score(test_true, test_pred, average='macro'))
        f.write('\n')
        f.write(classification_report(test_true, test_pred, target_names=['N', 'abN'], digits=5))
        f.write('\ntest false negative rate: %.5f\n' % test_fnr)
        f.write('sum  fnr: %.5f\n' % (val_fnr + test_fnr))
        f.write('\n\n')
    cnf_matrix = confusion_matrix(test_true, test_pred)
    plot_confusion_matrix(cnf_matrix, ['N', 'abN'], pic_name='cm2_test.jpg')

    # # show top 4 errors
    # errors = (val_pred != val_true)
    # val_file_num_errors = val_file_num[errors]
    # val_samples_errors = val_samples_mv[errors]
    # val_pred_matrix_errors = val_pred_matrix[errors]
    # val_true_errors = val_true[errors]
    # val_pred_errors = val_pred[errors]

    # pred_prob = np.amax(val_pred_matrix_errors, axis=1)
    # true_prob = np.diagonal(val_pred_matrix_errors[:, val_true_errors])
    # errors_prob_diff = pred_prob - true_prob
    # errors_indices = np.argsort(errors_prob_diff)
    # top4_errors = errors_indices[-4:]

    # show_errors(val_file_num_errors, val_samples_errors, top4_errors, val_true_errors, val_pred_errors)


if __name__ == '__main__':
    main()
