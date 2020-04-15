# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-03-24
"""
import os
import re
from collections import defaultdict
from constants import ROOT_PATH
from models.slot_filling.utils.lexicon import LexiconBuilder
from models.slot_filling.utils.trie import Trie

specific_words_file = os.path.join(ROOT_PATH, 'configs/specific_words.txt')


# 构建字典树
class Trees(object):
	def __init__(self, word_dict, lexi_trees):
		self.word_dict = word_dict
		self.lexi_trees = lexi_trees  # {'lexi_type':LexiconBuilder}

	@classmethod
	def build_trees(cls):
		lexi_trees = {}
		word_dict = defaultdict(list)
		with open(specific_words_file, 'r') as rf:
			for line in rf:
				line = re.split(r'[\s\t]+', line.strip())
				token, value = line[0], line[1]
				if token == 'word':
					continue
				else:
					word_dict[value].append(token)
		for lexi_type, tokens in word_dict.items():
			lb = LexiconBuilder(tokens, lexi_type)
			lexi_trees[lexi_type] = lb
		return cls(word_dict, lexi_trees)


if __name__ == '__main__':
	trees = Trees.build_trees()
	for k, v in trees.lexi_trees.items():
		print(v.search('灯'))
