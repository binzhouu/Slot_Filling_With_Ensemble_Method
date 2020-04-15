# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-03-31
"""
from datetime import datetime
import os
import pickle
import torch
import numpy as np
import re
import yaml
from pkuseg import pkuseg
from constants import ROOT_PATH
from models.slot_filling.utils.data import Data
from models.slot_filling.utils.data import Alphabet
from models.slot_filling.model.cnn_attn_lstm_crf import CnnAttnLstmCRF
from models.slot_filling.utils.functions import normalize_word, batch_char_sequence_labeling_with_word, predict_recover_label


yml_path = os.path.join(ROOT_PATH, 'models/slot_filling/conf/model.yaml')
dset_path = os.path.join(ROOT_PATH, 'saved_models/slot_filling/cnn_attn_lstm_crf/data/alphabet.dset')
model_dir = os.path.join(ROOT_PATH, 'saved_models/slot_filling/cnn_attn_lstm_crf/cnn_attn_lstm_crf.model')
user_dict = os.path.join(ROOT_PATH, 'lexicon/user_dict.txt')


class SlotModelV3(object):
	def __init__(self):
		with open(dset_path, 'rb') as rbf:
			char_alphabet = pickle.load(rbf)
			word_alphabet = pickle.load(rbf)
			feat_alphabet = pickle.load(rbf)
			label_alphabet = pickle.load(rbf)
			lexicon_alphabet = pickle.load(rbf)
			self.lexi_trees = pickle.load(rbf)
			char_alphabet_size = len(char_alphabet) + 1
			word_alphabet_size = len(word_alphabet) + 1
			feat_alphabet_size = len(feat_alphabet) + 1
			label_alphabet_size = len(label_alphabet) + 1
			lexicon_alphabet_size = len(lexicon_alphabet) + 1
		self.configs = self.read_configs(model_num=3)
		data = Data(
			[], [], [], [], [], [], char_alphabet_size, word_alphabet_size, feat_alphabet_size, label_alphabet_size, lexicon_alphabet_size)
		self.alphabet = Alphabet(word_alphabet, char_alphabet, feat_alphabet, label_alphabet, [], [], [], [], lexicon_alphabet, self.lexi_trees)
		self.model = CnnAttnLstmCRF(data, self.configs)
		assert self.configs['encoder_type'] == 'cnn_attn_lstm_crf'
		self.model.load_state_dict(torch.load(model_dir, map_location=self.configs['map_location']))
		self.model.eval()
		self.model.to(self.configs['device'])
		self.seg = pkuseg(user_dict=user_dict)

	def inference(self, text, intent, session_keep, previous_intent):
		# 如果存在多轮对话，且当前intent为空，取上一轮text的意图
		if session_keep and intent is None:
			intent = previous_intent
		if intent is None:  # label_alphabet中的None是str类型
			intent = 'None'
		instance_ids = []
		char_ids = []
		chars = list(text)
		# intent
		feat_ids = [self.alphabet.get_index(intent, key='intent')]
		# 字符
		for char in chars:
			char_id = self.alphabet.get_index(normalize_word(char), key='char')
			char_ids.append(char_id)
		# 词
		word_list = self.seg.cut(text)
		print('text: %s, word_list: %s' % (text, word_list))
		word_ids, word_feat_ids = [], []
		for word in word_list:
			lexi_feat = []
			for lexi_type, lb in self.lexi_trees.items():
				lexi_feat.append(lb.search(word))
			for n in range(len(lexi_feat)):
				if lexi_feat[n] is None or lexi_feat[n] == '_STEM_':
					lexi_feat[n] = 0
				else:
					lexi_feat[n] = 1
			lexi_feat = ''.join([str(i) for i in lexi_feat])
			word_feat_ids.append(self.alphabet.get_index(lexi_feat, 'lexicon'))

			word_id = self.alphabet.get_index(normalize_word(word), key='word')
			word_ids.append(word_id)
		instance_ids.append([char_ids, word_ids, feat_ids, word_feat_ids])
		batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, mask, \
			_, batch_lexi = batch_char_sequence_labeling_with_word(instance_ids, self.configs['gpu'], if_train=False, if_label=False)
		tag_seq = self.model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, mask, batch_lexi)
		pred_result = predict_recover_label(tag_seq, mask, self.alphabet, batch_charrecover)
		pred_result = list(np.array(pred_result).reshape(len(chars), ))
		result = self.slot_concat(chars, pred_result)
		return result

	@staticmethod
	def slot_concat(text_list, pred_result):
		pred_index = []
		start = 0
		for n in range(len(pred_result)):
			end = start + len(text_list[n])
			pred_index.append((start, end))
			start = end
		index_tmp, start, end = [], 0, 0
		entity_tmp = list()
		x = text_list
		y = pred_result
		entity = ''
		for n in range(len(y)):
			max_n = len(y) - 1
			if n < max_n:
				if y[n][0] == 'B' and len(entity) == 0:
					entity = y[n][2:] + ':' + x[n]
					start, end = pred_index[n][0], pred_index[n][-1]
				elif y[n][0] == 'I' and len(entity) != 0:
					entity += x[n]
					end = pred_index[n][-1]
				elif y[n][0] == 'O' and len(entity) != 0:
					entity_tmp.append(entity)
					index_tmp.append((start, end))
					entity = ''
				elif y[n][0] == 'B' and len(entity) != 0:  # B 连着 B 的情况
					entity_tmp.append(entity)
					index_tmp.append((start, end))
					entity = y[n][2:] + ':' + x[n]  # 重置entity
					start, end = pred_index[n][0], pred_index[n][-1]
			else:  # n == max_n
				if y[n][0] == 'B' and len(entity) == 0:
					entity = y[n][2:] + ':' + x[n]
					entity_tmp.append(entity)
					start, end = pred_index[n][0], pred_index[n][-1]
					index_tmp.append((start, end))
					entity = ''
				elif y[n][0] == 'B' and len(entity) != 0:  # B 连着 B 的情况
					entity_tmp.append(entity)
					index_tmp.append((start, end))
					entity = y[n][2:] + ':' + x[n]  # 重置entity
					start, end = pred_index[n][0], pred_index[n][-1]
					entity_tmp.append(entity)
					index_tmp.append((start, end))
				elif y[n][0] == 'I' and len(entity) != 0:
					entity += x[n]
					entity_tmp.append(entity)
					end = pred_index[n][-1]
					index_tmp.append((start, end))
				elif y[n][0] == 'O' and len(entity) != 0:
					entity_tmp.append(entity)
					index_tmp.append((start, end))
		result = []
		result_text = []
		for n in range(len(entity_tmp)):
			entity_list = []
			ent, idx_pair = entity_tmp[n], index_tmp[n]
			slot_label = re.split(r':', ent)[0]
			entity_list.append(slot_label)

			slot_name = re.split(r':', ent)[1]
			entity_list.append(slot_name)

			entity_list.append(idx_pair)
			result_text.append(entity_list)
		result.append(result_text)
		result = result[0] if result else result
		return result

	@staticmethod
	def read_configs(model_num):
		with open(yml_path, 'r') as rf:
			configs = yaml.load(rf, Loader=yaml.FullLoader)
		# 读取设备基本属性
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		gpu = True if device.type == 'cuda' else False
		map_location = 'cpu' if gpu is False else None
		configs.update({'device': device, 'gpu': gpu, 'map_location': map_location})
		# 读取model_num对应的模型超参数
		for k, v in configs['model'][model_num].items():
			configs[k] = v
		del configs['model']
		return configs


if __name__ == '__main__':
	texts = [
		'打开香薰机的灯', '洗衣机', '电视机', '客厅', '空调', '打开客厅空调', '晾衣架升高20%', '晾衣架的高度调高2米', '跟彩云城又尤为患者', '六月十日',
		'将客厅的空调调高2度', '我要看CCTV6', '设置空调为制热模式', '红色', '10秒', '到10秒', '调高一点', 'go go go', '晾衣架升高20%', '晾衣架调高点',
		'让晾衣架调高些', '帮我让晾衣架向上抬升些', '晾衣架升高2米', '把温度调高2', '温度调高2度', '湿度调高2', '温度降低3度', '湿度降低3', '把卧室灯设为调至50',
		'把空调调到25度', '调到1档', '把灯调到100%', '风扇调成1档', '空调调成23度', '雾量调到1档', '把客厅灯的颜色调为白色', '音量调到20%', '加湿器调到1档',
		'晾衣架升高2米']
	intents = [
		'turn_on', None, 'turn_off', 'set_model', 'increment_temperature', 'turn_on', 'increment_height', 'increment_height',
		None, None, 'increment_attribute', 'select_channel', 'set_attribute', 'set_color', 'fast_forward', 'fast_forward',
		'increment_attribute', None, 'increment_height', 'increment_height', 'increment_height', 'increment_height',
		'increment_height', 'increment_temperature', 'increment_temperature', 'increment_humidity', 'decrement_temperature',
		'decrement_humidity', 'set_brightness', 'set_temperature', 'set_level', 'set_attribute', 'set_level',
		'set_temperature', 'set_level', 'set_corlor', 'set_volume', 'set_level', 'increment_height']
	start = datetime.now()
	slot_filling = SlotModelV3()
	print('model init costs: %s' % (datetime.now() - start).total_seconds())
	for t, i in zip(texts, intents):
		start = datetime.now()
		res = slot_filling.inference(t, i, False, None)
		print('text: %s, intent: %s res: %s, time costs: %s' % (t, i, res, (datetime.now() - start).total_seconds()))
