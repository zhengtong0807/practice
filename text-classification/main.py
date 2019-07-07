import jieba
import pandas as pd
import os
import pprint

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from textcnn import create_model


def cut_stop(arr, label):
    """
    分词并去停用词
    """
    for sen in arr:
        sen = jieba.lcut(sen)
        sen_filter = []
        for word in sen:
            if word not in stop_words:
                sen_filter.append(word)

        X_total.append(' '.join(sen_filter))
        y_total.append(label)


if __name__ == "__main__":
    os.chdir('text-classification')
    X_total, y_total = [], []

    # 加载语料
    df0 = pd.read_csv('data/beilaogongda.csv').dropna()
    df1 = pd.read_csv('data/beilaopoda.csv').dropna()
    df2 = pd.read_csv('data/beierzida.csv').dropna()
    df3 = pd.read_csv('data/beinverda.csv').dropna()

    # 加载停用词典
    with open('data/stop_words.txt', encoding='utf8') as f:
        stop_words = [line.strip() for line in f.readlines()]

    # 生成数据
    cut_stop(df0.segment, 0)
    cut_stop(df1.segment, 1)
    cut_stop(df2.segment, 2)
    cut_stop(df3.segment, 3)
    # pprint.pprint(X_total[:5])
    # pprint.pprint(y_total[:5])

    tok = Tokenizer()
    tok.fit_on_texts(X_total)
    vocabulary = len(tok.word_index)
    # print(vocabulary)  # 665
    X_total = tok.texts_to_sequences(X_total)
    # pprint.pprint(X_total[:5])
    max_len = max([len(sen) for sen in X_total])
    # print(max_len)  # 494
    X_total = pad_sequences(X_total,
                            maxlen=max_len,
                            padding='post',
                            truncating='post')

    # 生成训练集、测试集
    test_size = .2
    num_class = 4
    X_train, X_test, y_train, y_test = train_test_split(X_total, y_total,
                                                        test_size=test_size,
                                                        random_state=666,
                                                        stratify=y_total)
    y_train = to_categorical(y_train, num_class)
    y_test = to_categorical(y_test, num_class)

    model = create_model(max_len, vocabulary, num_class)
    batch_size = 128
    epochs = 10
    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, y_test))
