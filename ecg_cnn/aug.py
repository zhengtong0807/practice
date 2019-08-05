"""data augmentation"""

import numpy as np

REST = 300  # 1s

a_step = REST // 5  # 60
o_step = REST // 2  # 150
n_step = REST // 20  # 15


def wgn(x, snr):
    snr = 10**(snr / 10.0)
    xpower = np.sum(x**2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def augmentation(value, label):
    samples = []
    labels = []

    length = len(value)
    window = length - REST

    if label == 1:  # A
        for i in range(6):
            sample = value[i * a_step:window + i * a_step]
            # sample = sample + wgn(sample, 30)  # 增加了30dBz信噪比噪声的信号
            samples.append(sample)
            labels.append(label)

    elif label == 2:  # O
        sample = value[o_step:window + o_step]
        # sample = value + wgn(value, 30)
        samples.append(sample)
        labels.append(label)

    elif label == 3:  # ~
        for i in range(20):
            sample = value[i * n_step:window + i * n_step]
            # sample = sample + wgn(sample, 30)
            samples.append(sample)
            labels.append(label)

    return samples, labels
