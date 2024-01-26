# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('zh_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def calculate_distances(sentence, target_word):
    words = sentence.split()
    distances = []
    matrix = np.zeros((len(words), len(words))).astype('float32')
    target_index = words.index(target_word)
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

def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1
    # print(matrix)
    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 3):
        sentence = lines[i].strip()
        # print(sentence)
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("aspect")]
        aspect = lines[i + 1].lower().strip()
        # print(target_word)
        target_word = aspect.split()[0]
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        matrix2 = calculate_distances(text_left+' '+aspect+' '+text_right, target_word)
        matrix2 = normalize_matrix(matrix2)
        adj_matrix = adj_matrix + matrix2
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
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
