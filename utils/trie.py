# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-03-23
"""
import logging


logger = logging.getLogger(__file__)


class Trie(object):
	def __init__(self, words=None, values=None):
		self.root = {}
		if words:
			assert isinstance(words, list)
			if values:
				assert len(words) == len(values)
				for word, value in zip(words, values):
					self.add(word, value)
			else:
				for word in words:
					self.add(word, value='_FALSE_')  # '_FALSE_'是叶子节点的标志

	def add(self, word, value='_FALSE_'):
		try:
			assert isinstance(word, str)
		except AssertionError:
			logger.info('Class Trie add(): input parameter type must be string')
		else:
			node = self.root
			for c in word:
				if c not in node:
					node[c] = {}  # {'a': {}}
				node = node[c]  # c存在，node={'涛': {'_LEAF_': '_FALSE_'};c不存在,node='a'对应的{}
			node['_LEAF_'] = value  # 表示一个词的结束

	def delete(self, word):
		try:
			assert isinstance(word, str)
		except AssertionError:
			logger.info('Class Trie delete(): input parameter type must be string')
			return
		node = self.root
		catch_flag = False
		del_node = node
		del_key = word[0]
		for c in word:
			if c not in node:
				logger.info('word %r is not in the dictionary' % word)
				return
			if catch_flag:
				del_key = c  # char不是word[0]时，del_key跟着转移
			node = node[c]
			edges = len(node)  # del_word的首个字对应的dict，的数量
			if edges > 1:
				del_node = node
				if c == word[-1]:
					del_key = '_LEAF_'
				else:
					catch_flag = True
			else:
				catch_flag = False
			print('del_key:', del_key)
			print('node:', node)
			print('del_node:', del_node)
		if '_LEAF_' in node:
			del_node.pop(del_key)
			logger.info('SUCCESS: word %r is deleted' % word)
		else:
			logger.info('WARNING: word %r is not in the dictionary' % word)
			return

	def search(self, word, value_flag=False, verbose=False):
		try:
			assert isinstance(word, str)
		except AssertionError:
			print('WARNING: Class Trie search(): input parameter type must be string')
			return None
		node = self.root
		for c in word:
			if c not in node:
				if verbose:  # 是否显示错误信息
					print('FAIL: word %r is not in the lexicon' % word)
				return None
			node = node[c]  # 切换到首字对应的dict
		if '_LEAF_' in node:
			if value_flag:  # 显示value或者默认'_TRUE_'
				return node['_LEAF_']
			else:
				return '_TRUE_'
		elif node:
			return '_STEM_'
		else:
			return None

	def get_value(self, word, verbose=False):
		value_flag = True
		return self.search(word, value_flag, verbose)

	def show(self, num=2):
		try:
			isinstance(num, int)
			assert num <= len(self.root)
		except TypeError:
			print('WARNING: class Trie show(): input parameter type is integer')
			num = len(self.root)
		except AssertionError:
			print('WARNING: class Trie show(): input number is larger than trie size')
			num = len(self.root)
		if num < len(self.root):
			cnt = 0
			new_dict = {}
			for key, val in self.root.items():
				new_dict[key] = val
				cnt += 1
				if cnt >= num:  # num：控制展示的字典数量
					break
		else:
			new_dict = self.root

		import json
		out = json.dumps(new_dict, sort_keys=True, indent=4, ensure_ascii=False)
		print(out)

	def __str__(self):
		return repr(self.root)


if __name__ == '__main__':
	obj = Trie()
	obj.add('空调', 'han')
	obj.add('他', 'han')
	obj.add('我们', 'han')
	obj.show(4)
	obj.add('我们', 'sho')
	obj.show(4)
