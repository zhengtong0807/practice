import jieba
from jieba.analyse import extract_tags
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import os

os.chdir('TF-IDF')  # 修改当前工作目录
topK = 3  # 返回 TF-IDF 值最大的关键词个数


def filter_stop_words(sen, stop_words):
    '''
    对分词后的文本去停用词
    Parameter:
        - sen: 1-d list，分词后的文本
        - stop_words: 1-d list，停用词典
    '''
    sen_filter = []
    for word in sen:
        if word not in stop_words:
            sen_filter.append(word)

    return sen_filter


# **********jieba提取关键词**********
sen = '自然语言处理是人工智能和语言学领域的分支学科，此领域探讨如何处理及运用自然语言，包括多方面和步骤。'
print('  jieba extract:', extract_tags(sen, topK=topK))  # ['自然语言', '领域', '处理']


# **********sklearn提取关键词**********
corpus = [  # 语料
    '自然语言处理是人工智能和语言学领域的分支学科，此领域探讨如何处理及运用自然语言，包括多方面和步骤。',
    '计算机视觉是一门研究如何使机器“看”的科学，用摄影机和计算机代替人眼对目标进行识别、跟踪和测量。',
    '机器学习是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。'
]
corpus = [jieba.lcut(sen) for sen in corpus]
with open('stop_words.txt', encoding='utf8') as f:
    stop_words = [line.strip() for line in f.readlines()]
corpus = [' '.join(filter_stop_words(sen, stop_words)) for sen in corpus]

cvec = CountVectorizer()
cvec.fit_transform(corpus)
feature_words = cvec.get_feature_names()
feature_words = np.array(feature_words)

tvec = TfidfVectorizer()
tvec = tvec.fit_transform(corpus)
first_sen = tvec.toarray()[0]
max_indices = np.argsort(-first_sen)[:topK]
print('sklearn extract:', feature_words[max_indices])  # ['自然语言' '领域' '语言学']
