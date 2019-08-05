"""data process"""

import os

# import xml.etree.ElementTree as ET  # xml
import scipy.io as sio  # mat

import numpy as np
from sklearn import preprocessing
from keras import utils
from keras.preprocessing import sequence

# from aug import augmentation

LENGTH = 30  # s
FREQUENCY = 500
PAD_LENGTH = LENGTH * FREQUENCY  # 15000

# DATA_DIR = '/home/welheart/Datasets/20180716'  # xml
DATA_DIR = '/home/welheart/wxs/dataset'  # mat

RHYTHM_LIST = ['N', 'A', 'F', 'S', 'Fz']
NUM_CLASSES = len(RHYTHM_LIST)


def data_process(files, labels, is_aug=False):
    all_files = None

    for file in files:
        # file_path = os.path.join(DATA_DIR, file + '.xml')
        file_path = os.path.join(DATA_DIR, file + '.mat')

        # tree = ET.parse(file_path)
        # root = tree.getroot()
        # namespace = root.tag[:-12]  # root.tag : {urn:hl7-org:v3}AnnotatedECG
        # one_file = []
        # for idx, lead in enumerate(root.iter(namespace + 'digits')):
        #     # if idx == 1:  # only select lead II
        #     #     one_file.append([int(i) for i in lead.text.strip().split()])
        #     one_file.append([int(i) for i in lead.text.strip().split()])

        one_file = None
        for key, value in sio.loadmat(file_path).items():
            # # 1 lead
            # if key == 'MDC_ECG_LEAD_II':
            #     one_file = value

            # 12 leads
            if one_file is None:
                one_file = value
            else:
                one_file = np.concatenate((one_file, value))

        one_file = preprocessing.scale(one_file, axis=1)
        one_file = sequence.pad_sequences(one_file, 
                                          maxlen=PAD_LENGTH, 
                                          dtype='float32', 
                                          padding='post', 
                                          truncating='post')
        one_file = one_file.T[None, :, :]

        if all_files is None:
            all_files = one_file
        else:
            all_files = np.concatenate((all_files, one_file))

    all_labels = utils.to_categorical(labels, NUM_CLASSES)

    assert all_files.shape[0] == all_labels.shape[0]

    return all_files, all_labels


'''read all val data (mat):
read 128 files, Time used: 3.8803s  (31.69m 7.92m)
read 256 files, Time used: 14.5972s
read 384 files, Time used: 32.0352s
read 512 files, Time used: 56.3645s
read 640 files, Time used: 87.9645s
'''

'''read all val data (mat, one):
read 128 files, Time used: 0.2669s  (2.18m 32.70s)
read 256 files, Time used: 0.6915s
read 384 files, Time used: 1.7266s
read 512 files, Time used: 3.7663s
read 640 files, Time used: 6.4483s
'''

'''read all val data (xml):
read 128 files, Time used: 6.0709s
read 256 files, Time used: 19.0702s
read 384 files, Time used: 38.8606s
read 512 files, Time used: 65.5315s
read 640 files, Time used: 99.1298s
'''

'''read all val data (xml, one):
read 128 files, Time used: 0.6464s
read 256 files, Time used: 1.2348s
read 384 files, Time used: 2.5666s
read 512 files, Time used: 4.9223s
read 640 files, Time used: 7.9222s
'''

'''
MDC_ECG_LEAD_I
MDC_ECG_LEAD_II
MDC_ECG_LEAD_III
MDC_ECG_LEAD_AVR
MDC_ECG_LEAD_AVL
MDC_ECG_LEAD_AVF
MDC_ECG_LEAD_V1
MDC_ECG_LEAD_V2
MDC_ECG_LEAD_V3
MDC_ECG_LEAD_V4
MDC_ECG_LEAD_V5
MDC_ECG_LEAD_V6
'''
