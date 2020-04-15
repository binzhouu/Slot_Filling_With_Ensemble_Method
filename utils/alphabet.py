# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-11-11
"""
import os
import pickle
import ast
from collections import Counter
from collections import defaultdict
import logging
import yaml
from models.slot_filling.utils.functions import normalize_word
from constants import ROOT_PATH

logger = logging.getLogger(__name__)
head_path, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))
# data_file = os.path.join(head_path, 'data/output/intent_corpus_v2.txt')
# data_file = os.path.join(head_path, 'data/output/slot_corpus.txt')
data_file = os.path.join(head_path, 'data/output/demo.txt')

with open(os.path.join(ROOT_PATH, 'models/slot_filling/conf/data.yaml'), 'r') as rf:
	data_configs = yaml.load(rf, Loader=yaml.FullLoader)
model_num = data_configs['model_num']
dset_path = data_configs['model'][model_num]['dset_path']
dset_path = os.path.join(ROOT_PATH, dset_path)


class Alphabet(object):
	def __init__(self, word_alphabet, char_alphabet, feat_alphabet, label_alphabet, words, chars, feats, labels, lexicon_alphabet, lexi_trees):
		self.word_alphabet, self.feat_alphabet, self.label_alphabet, self.char_alphabet = \
			word_alphabet, feat_alphabet, label_alphabet, char_alphabet
		self.lexicon_alphabet = lexicon_alphabet
		self.lexi_trees = lexi_trees
		self.words, self.chars, self.feats, self.labels = words, chars, feats, labels
		self.UNKNOWN = '/unk'
		self.write_alphabet()

	@classmethod
	def build_alphabet(cls, input_file, lexi_trees):
		with open(input_file, 'r') as rf:
			intent_corpus = rf.readlines()
			words, chars, feats, labels, lexicons = [], [], [], [], []
			word_alphabet, char_alphabet, feat_alphabet, label_alphabet, lexicon_alphabet = {}, {}, {}, {}, {}
			for i in intent_corpus:
				# print(i)
				line = ast.literal_eval(i)
				char, label, word, feat = line['char'], line['char_label'], line['word'], line['intent']
				word = list(map(lambda x: normalize_word(x), word))
				char = list(map(lambda x: normalize_word(x), char))
				# 增加字典树搜索得到的特征
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
					if lexi_feat not in lexicons: lexicons.append(lexi_feat)
				words.extend(word)
				chars.extend(char)
				labels.extend(label)
				feats.append(feat)
		words = list(Counter(words).keys())
		chars = list(Counter(chars).keys())
		labels = list(Counter(labels).keys())
		feats = list(Counter(feats).keys())

		words = ['/unk'] + words
		chars = ['/unk'] + chars
		feats = ['/unk'] + feats

		lexicons = ['/unk'] + lexicons

		for i, v in enumerate(words): word_alphabet[v] = i + 1
		for i, v in enumerate(chars): char_alphabet[v] = i + 1
		for i, v in enumerate(labels): label_alphabet[v] = i + 1
		for i, v in enumerate(feats): feat_alphabet[v] = i + 1

		for i, v in enumerate(lexicons): lexicon_alphabet[v] = i + 1

		logger.info('intent nums: %s, slot nums: %s' % (len(feat_alphabet), len(label_alphabet)))
		return cls(word_alphabet, char_alphabet, feat_alphabet, label_alphabet, words, chars, feats, labels, lexicon_alphabet, lexi_trees)

	def write_alphabet(self):
		if not os.path.exists(dset_path):
			f = open(dset_path, 'wb')
			pickle.dump(self.char_alphabet, f)
			pickle.dump(self.word_alphabet, f)
			pickle.dump(self.feat_alphabet, f)
			pickle.dump(self.label_alphabet, f)
			pickle.dump(self.lexicon_alphabet, f)
			pickle.dump(self.lexi_trees, f)
			f.close()

	def get_index(self, instance, key):
		if key == 'char':
			try:
				return self.char_alphabet[instance]
			except KeyError:
				return self.char_alphabet[self.UNKNOWN]
		if key == 'word':
			try:
				return self.word_alphabet[instance]
			except KeyError:
				return self.word_alphabet[self.UNKNOWN]
		if key == 'intent':
			try:
				return self.feat_alphabet[instance]
			except KeyError:
				return self.feat_alphabet[self.UNKNOWN]
		if key == 'lexicon':
			try:
				return self.lexicon_alphabet[instance]
			except KeyError:
				return self.lexicon_alphabet[self.UNKNOWN]
		if key in ['char_label', 'word_label']:
			return self.label_alphabet[instance]

	def get_instance(self, index, key):
		if key == 'label':
			reversed_dict = {v: k for k, v in self.label_alphabet.items()}
			return reversed_dict[index]


if __name__ == '__main__':
	alphabet = Alphabet.build_alphabet(data_file)
