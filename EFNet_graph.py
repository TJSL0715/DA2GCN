# -*- coding: utf-8 -*-

import numpy as np
import pickle


def load_EFNet_word():
    path = './Chinese/汉语情感词极值表.txt'
    EFNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, EF = line.split('\t')
        EFNet[word] = EF
    fp.close()
    return EFNet


def dependency_ad_matrix(text, aspect, EFNet):
    word_list = text.split()
    seq_len = len(word_list)
    # print(seq_len)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    for i in range(seq_len):
        word = word_list[i]
        if word in EFNet:
            EF = abs(float(EFNet[word]))
        else:
            EF = 0
        if word in aspect:
            EF += 1.0
        for j in range(seq_len):
            matrix[i][j] += EF
            matrix[j][i] += EF
    for i in range(seq_len):
        if matrix[i][i] == 0:
            matrix[i][i] = 1
    # print(matrix)
    return matrix


def calculate_distances(sentence, aspect):
    # aspect = aspect.replace(" ", "")
    words = sentence.split()
    # print(words)
    matrix = np.zeros((len(words), len(words))).astype('float32')
    target_index = words.index(aspect)
    for i, word in enumerate(words):
        distance = abs(target_index - i)
        # distances.append(distance)
        matrix[i][i] = distance
    # print(matrix)
    # print(words)
    return matrix


def normalize_matrix(matrix):
    # 找到矩阵中的最大元素
    max_value = np.max(matrix)

    # 遍历矩阵的每个元素，将非零元素除以最大元素
    normalized_matrix = np.where(matrix != 0, 1 - matrix / 2 / max_value, matrix)

    return normalized_matrix

def process(filename):
    EFNet = load_EFNet_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    # target_word = "$T$"
    fout = open(filename+'.ef_graph', 'wb')
    for i in range(0, len(lines), 3):
        sentence = lines[i].strip()
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("aspect")]
        aspect = lines[i + 1].lower().strip()
        target_word = aspect.split()[0]
        matrix2 = calculate_distances(text_left+' '+aspect+' '+text_right, target_word)
        matrix2 = normalize_matrix(matrix2)
        adj_matrix = dependency_ad_matrix(text_left+' '+aspect+' '+text_right, aspect, EFNet)
        adj_matrix = adj_matrix + matrix2
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('done !!!', filename)
    fout.close()


if __name__ == '__main__':
    # process('./datasets/acl-14-short-data/train.raw')
    # process('./datasets/acl-14-short-data/test.raw')
    # process('./datasets/semeval14/restaurant_train.raw')
    # process('./datasets/semeval14/restaurant_test.raw')
    # process('./datasets/semeval14/laptop_train.raw')
    # process('./datasets/semeval14/laptop_test.raw')
    # process('./datasets/semeval15/restaurant_train.raw')
    # process('./datasets/semeval15/restaurant_test.raw')
    # process('./datasets/semeval16/restaurant_train.raw')
    # process('./datasets/semeval16/restaurant_test.raw')
    process('./datasets/mooc/train.raw')
    process('./datasets/mooc/test.raw')
    process('./datasets/douban/train.raw')
    process('./datasets/douban/test.raw')
