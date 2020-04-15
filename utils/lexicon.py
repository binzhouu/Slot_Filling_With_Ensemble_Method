# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-03-24
"""
import os
import re
from collections import defaultdict
from models.slot_filling.utils.trie import Trie
from constants import ROOT_PATH


class LexiconBuilder(object):
	def __init__(self, tokens, value):
		self.lexicon = Trie()
		self.build(tokens, value)  # 每次都会初始化pkl

	# def _build(self):
	# 	cnt = 0
	# 	with open(self._path, 'r', encoding='utf-8') as rf:
	# 		for num, line in enumerate(rf, 1):
	# 			line = line.strip()
	# 			if line:
	# 				[token] = line.split()
	# 				self.lexicon.add(token, value)
	# 				cnt += 1

	def build(self, tokens, value):
		"""

		:param tokens: word_list:['电视机', '电视', ...]
		:param value: word_onto:'tool','show_name',...
		:return:
		"""
		for token in tokens:
			self.lexicon.add(token, value)

	def show(self, num):
		return self.lexicon.show(num)

	def search(self, word):
		return self.lexicon.search(word, value_flag=True, verbose=False)

	# def __str__(self):  # 返回所有字典绝对路径
	# 	return 'Lexicon set=%r' % self._path


if __name__ == '__main__':
	pass
