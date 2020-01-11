import jieba
from jieba.analyse import extract_tags
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# TODO: jieba加载行业词典

TOPK = 3  # 返回 TF-IDF 值最大的关键词个数
WITH_WEIGHT = False  # 是否带权重返回


def filter_stop_words(sen, stop_words):
    """
    对分词后的文本去停用词
    Parameter:
        - sen: 1-d list，分词后的文本
        - stop_words: 1-d list，停用词典
    """
    sen_filter = []
    for word in sen:
        if word not in stop_words:
            sen_filter.append(word)

    return sen_filter


def sklearn_tfidf_tags(corpus, topK=3, withWeight=False):
    corpus = [jieba.lcut(sen) for sen in corpus]
    with open('stop_words.txt', encoding='utf8') as f:
        stop_words = [line.strip() for line in f.readlines()]
    corpus = [' '.join(filter_stop_words(sen, stop_words)) for sen in corpus]

    tvec = TfidfVectorizer()
    X = tvec.fit_transform(corpus)
    feature_words = np.array(tvec.get_feature_names())

    results = []
    for item in X.toarray():
        max_indices = np.argsort(-item)[:topK]
        if withWeight:
            results.append(list(zip(feature_words[max_indices], item[max_indices])))
        else:
            results.append(feature_words[max_indices].tolist())

    return results


if __name__ == '__main__':
    # **********jieba提取关键词************ #
    sen = '自然语言处理是人工智能和语言学领域的分支学科，此领域探讨如何处理及运用自然语言，包括多方面和步骤。'
    print('\n  jieba extract:', extract_tags(sen, topK=TOPK, withWeight=WITH_WEIGHT))  # ['自然语言', '领域', '处理']

    # **********sklearn提取关键词********** #
    corpus = [  # 语料
        '自然语言处理是人工智能和语言学领域的分支学科，此领域探讨如何处理及运用自然语言，包括多方面和步骤。',
        '计算机视觉是一门研究如何使机器“看”的科学，用摄影机和计算机代替人眼对目标进行识别、跟踪和测量。',
        '机器学习是一门多领域交叉学科，涉及概率论、统计学、逼近论、凸分析、算法复杂度理论等多门学科。'
    ]
    print('\nsklearn extract:', sklearn_tfidf_tags(corpus, topK=TOPK, withWeight=WITH_WEIGHT))  # ['自然语言' '领域' '语言学']
