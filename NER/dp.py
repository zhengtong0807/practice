"""
数据预处理
"""


def data_process():
    zh_punctuation = ['，', '。', '？', '；', '！', '……']

    with open('data/BosonNLP_NER_6C_process.txt', 'w', encoding='utf8') as fw:
        with open('data/BosonNLP_NER_6C.txt', encoding='utf8') as fr:
            for line in fr.readlines():
                line = ''.join(line.split()).replace('\\n', '')  # 去除文本中的空字符

                i = 0
                while i < len(line):
                    word = line[i]

                    if word in zh_punctuation:
                        fw.write(word + '/O')
                        fw.write('\n')
                        i += 1
                        continue

                    if word == '{':
                        i += 2
                        temp = ''
                        while line[i] != '}':
                            temp += line[i]
                            i += 1
                        i += 2

                        type_ne = temp.split(':')
                        etype = type_ne[0]
                        entity = type_ne[1]
                        fw.write(entity[0] + '/B_' + etype + ' ')
                        for item in entity[1:]:
                            fw.write(item + '/I_' + etype + ' ')
                    else:
                        fw.write(word + '/O ')
                        i += 1


if __name__ == '__main__':
    data_process()
