from pprint import pprint
import jieba
from gensim.models import Word2Vec
import logging

from utils import small_corpus, stop_words

jieba.add_word('机器学习')
jieba.add_word('深度学习')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    # 分词并去停用词
    corpus = [
        [word for word in jieba.lcut(text) if word not in stop_words]
        for text in small_corpus
    ]

    # 可以选择过滤一下高频或低频词（去掉高频词是因为没有特殊性，去掉低频词是因为没有普适性）
    filter_freq = False  # 语料太小，不过滤了
    if filter_freq:
        freq = {}
        for text in corpus:
            for word in text:
                freq[word] = freq.get(word, 0) + 1
        corpus = [
            [word for word in text if freq[word] > 1]  # 去掉仅出现 1次 的词
            for text in corpus
        ]
        # pprint(freq)

    print(*corpus, sep=',\n')

    # 训练词向量并保存
    model = Word2Vec(corpus, min_count=0)
    model.save('model')
    model.wv.save_word2vec_format('word2vec.txt')

    # 加载模型
    model = Word2Vec.load('model')
    pprint(model.wv.index2word, compact=True)  # 获得所有的词汇
    print(model.wv['机器学习'])  # 打印某个词对应的词向量
    print(model.similarity('机器学习', '深度学习'))  # 计算词之间的相似度


if __name__ == '__main__':
    # pprint(small_corpus)
    # pprint(stop_words[:10])
    main()
