# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-11-12
"""
import os
import pickle
from models.slot_filling.utils.alphabet import Alphabet
import ast
import numpy as np
import random
import logging
from models.slot_filling.utils.functions import generate_char, normalize_word
from constants import ROOT_PATH


head_path, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
# data_file = os.path.join(head_path, 'data/output/intent_corpus_v3.txt')
# data_file = os.path.join(head_path, 'data/output/slot_corpus.txt')
data_file = os.path.join(head_path, 'data/output/demo.csv')
logger = logging.getLogger(__name__)


class Data(object):
    def __init__(self, train_texts, dev_texts, test_texts, train_ids, dev_ids, test_ids,
                 char_alphabet_size, word_alphabet_size, feat_alphabet_size, label_alphabet_size, lexicon_alphabet_size):

        self.train_texts, self.dev_texts, self.test_texts, self.train_ids, self.dev_ids, self.test_ids =\
            train_texts, dev_texts, test_texts, train_ids, dev_ids, test_ids
        self.lexicon_alphabet_size = lexicon_alphabet_size
        self.char_alphabet_size, self.word_alphabet_size, self.feat_alphabet_size, self.label_alphabet_size =\
            char_alphabet_size, word_alphabet_size, feat_alphabet_size, label_alphabet_size

    @classmethod
    def read_instance(cls, alphabet, input_file, lexi_trees, read_type='word'):
        texts, ids = [], []
        # char, char_id = [], []
        with open(input_file, 'r') as rf:
            intent_corpus = rf.readlines()
            for i in intent_corpus:
                line = ast.literal_eval(i)
                char, word, feat, word_label, char_label = line['char'], line['word'], line['intent'], line['word_label'], line['char_label']
                # 加入字典树搜索的特征：
                word_feat, word_feat_id = [], []
                for w in word:
                    lexi_feat = []
                    for lexi_type, lb in lexi_trees.items():
                        lexi_feat.append(lb.search(w))
                    for n in range(len(lexi_feat)):
                        if lexi_feat[n] is None or lexi_feat[n] == '_STEM_':
                            lexi_feat[n] = 0
                        else:
                            lexi_feat[n] = 1
                    lexi_feat = ''.join([str(i) for i in lexi_feat])
                    word_feat.append(lexi_feat)
                    word_feat_id.append(alphabet.get_index(lexi_feat, 'lexicon'))
                texts.append([char, word, feat, char_label, word_feat])

                if not isinstance(feat, list): feat = [feat]
                # normalized处理数字
                char = list(map(lambda x: normalize_word(x), char))
                word = list(map(lambda x: normalize_word(x), word))
                # 字需要用函数处理一下
                char, char_id = generate_char(char, word, alphabet)
                # char_id = [alphabet.get_index(c, 'char') for c in char]
                word_id = [alphabet.get_index(w, 'word') for w in word]
                feat_id = [alphabet.get_index(f, 'intent') for f in feat]
                # 将label替换为字符级别
                if read_type == 'char':
                    label_id = [alphabet.get_index(l, 'char_label') for l in char_label]
                else:
                    label_id = [alphabet.get_index(l, 'word_label') for l in word_label]
                ids.append([char_id, word_id, feat_id, label_id, word_feat_id])

        indexes = list(range(len(ids)))
        random.seed(43)
        random.shuffle(indexes)
        texts = [texts[i] for i in indexes]
        ids = [ids[i] for i in indexes]
        logger.info('indexes: %s' % indexes[:10])

        n = int(len(ids) * 0.1)  # 抽样比例
        dev_texts, dev_ids = texts[:n], ids[:n]
        test_texts, test_ids = texts[n:2*n], ids[n:2*n]
        train_texts, train_ids = texts[2*n:], ids[2*n:]

        # 将test集写入文件，待模型训练完成后做验证
        # with open(os.path.join(ROOT_PATH, 'models/slot_filling/data/output/test_texts.pkl'), 'wb') as wbf:
        #     pickle.dump(test_texts, wbf)

        char_alphabet_size, word_alphabet_size, feat_alphabet_size, label_alphabet_size = \
            len(alphabet.char_alphabet) + 1, len(alphabet.word_alphabet) + 1, len(alphabet.feat_alphabet) + 1, \
            len(alphabet.label_alphabet) + 1
        lexicon_alphabet_size = len(alphabet.lexicon_alphabet) + 1
        logger.info('train_size:%s, dev_size:%s, test_size:%s' % (len(train_texts), len(dev_texts), len(test_texts)))
        return cls(train_texts, dev_texts, test_texts, train_ids, dev_ids, test_ids,
                   char_alphabet_size, word_alphabet_size, feat_alphabet_size, label_alphabet_size, lexicon_alphabet_size)


if __name__ == '__main__':
    alphabet = Alphabet.build_alphabet(data_file)
    data = Data.read_instance(alphabet, data_file, read_type='char')
