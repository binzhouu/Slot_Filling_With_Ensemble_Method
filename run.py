# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-03-04
"""
import os
import sys
ROOT_PATH = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
sys.path.append(ROOT_PATH)
import yaml
import numpy as np
import torch
import logging
import logging.config
import torch.optim as optim
import datetime
import time
from models.slot_filling.utils.functions import batch_char_sequence_labeling, predict_check, evalute_model, \
	batch_char_sequence_labeling_with_word
from tensorboardX import SummaryWriter
from models.slot_filling.model.bilstm_crf import BiLstmCrf
from models.slot_filling.model.attn_bilstm_crf import AttnBiLstmCRF
from models.slot_filling.model.cnn_attn_lstm_crf import CnnAttnLstmCRF
from models.slot_filling.model.multiply_bilstm_crf import MulBilstmCRF
from models.slot_filling.utils.data import Data
from models.slot_filling.utils.alphabet import Alphabet
from models.slot_filling.utils.trees import Trees

logger = logging.getLogger(__name__)
logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
writer = SummaryWriter('./tensorboard_log')

# data_file = os.path.join(ROOT_PATH, 'models/slot_filling/data/output/intent_corpus_v3.txt')
# data_file = os.path.join(ROOT_PATH, 'models/slot_filling/data/output/demo.txt')
# data_file = os.path.join(ROOT_PATH, 'models/slot_filling/data/output/slot_corpus.txt')
data_file = os.path.join(ROOT_PATH, 'models/data_source/Notebook/niudi.txt')
yml_path = os.path.join(ROOT_PATH, 'models/slot_filling/conf/model.yaml')


class Run(object):
	def __init__(self, configs):
		self.configs = configs
		self.encoder_type = self.configs['encoder_type']
		self.trees = Trees.build_trees()  # 构建字典树

	def train(self, data, alphabet):
		if self.encoder_type == 'bilstm_crf':
			model = BiLstmCrf(data, self.configs)
		elif self.encoder_type == 'attn_bilstm_crf':
			model = AttnBiLstmCRF(data, self.configs)
		elif self.encoder_type == 'cnn_attn_lstm_crf':
			model = CnnAttnLstmCRF(data, self.configs)
		elif self.encoder_type == 'mul_bilstm_crf':
			model = MulBilstmCRF(data, self.configs)
		else:
			print('No model select')
			return
		logger.info("configs: %s" % self.configs)
		# optimizer = optim.SGD(
		# 	model.parameters(), lr=model.configs['lr'], momentum=model.configs['momentum'], weight_decay=model.configs['l2'])
		optimizer = optim.Adam(model.parameters(), lr=self.configs['lr'], weight_decay=self.configs['l2'])
		if self.configs['gpu']:
			model = model.cuda()
		best_dev = -10
		last_improved = 0
		logger.info('train start: %s' % datetime.datetime.now())
		for idx in range(self.configs['epoch']):
			epoch_start = time.time()
			temp_start = epoch_start
			logger.info('Epoch : %s/%s' % (idx, self.configs['epoch']))
			optimizer = self.lr_decay(optimizer, idx, self.configs['lr_decay'], self.configs['lr'])

			sample_loss = 0
			total_loss = 0
			right_token = 0
			whole_token = 0
			logger.info('first input word list: %s, %s' % (data.train_texts[0][1], data.train_ids[0][1]))

			model.train()
			model.zero_grad()
			batch_size = self.configs['batch_size']
			train_num = len(data.train_ids)
			total_batch = train_num // batch_size + 1
			logger.info('total_batch: %s' % total_batch)

			for batch_id in range(total_batch):
				start = batch_id * batch_size
				end = (batch_id + 1) * batch_size
				if end > train_num:
					end = train_num
				instance = data.train_ids[start: end]
				if not instance:
					continue
				if self.encoder_type == 'cnn_attn_lstm_crf':
					batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, mask,\
						batch_label, batch_lexi = batch_char_sequence_labeling_with_word(instance, self.configs['gpu'])
					loss, tag_seq = model(
						batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, mask, batch_lexi, batch_label)
				else:
					batch_char, batch_features, batch_charlen, batch_charrecover, mask, batch_label = \
						batch_char_sequence_labeling(instance, self.configs['gpu'])
					loss, tag_seq = model(batch_char, batch_features, batch_charlen, batch_charrecover, mask, batch_label)
				# print(batch_id, loss)
				right, whole = predict_check(tag_seq, batch_label, mask)
				right_token += right
				whole_token += whole
				sample_loss += loss.item()
				total_loss += loss.item()

				if end % (batch_size * 10) == 0:
					temp_time = time.time()
					temp_cost = temp_time - temp_start
					temp_start = temp_time
					logger.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
						end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
					if sample_loss > 1e8 or str(sample_loss) == 'nan':
						raise ValueError(
							"ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
					sample_loss = 0

				loss.backward()
				optimizer.step()
				model.zero_grad()
			temp_time = time.time()
			temp_cost = temp_time - temp_start
			logger.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
				end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
			epoch_finish = time.time()
			epoch_cost = epoch_finish - epoch_start
			logger.info("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
				idx, epoch_cost, train_num / epoch_cost, total_loss))
			if total_loss > 1e8 or str(total_loss) == 'nan':
				raise ValueError("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")

			writer.add_scalar('Train_loss', total_loss, idx)
			speed, acc, p, r, f, _, _ = evalute_model(data, model, "dev", self.configs, alphabet, self.encoder_type)
			dev_finish = time.time()
			dev_cost = dev_finish - epoch_finish
			current_score = f
			writer.add_scalar('Dev_f1_score', current_score, idx)
			logger.info("Epoch: %s, Loss: %s, Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
				idx, total_loss, dev_cost, speed, acc, p, r, f))

			if current_score > best_dev:
				model_name = self.configs['model_path'] + '.model'
				model_name = os.path.join(ROOT_PATH, model_name)
				torch.save(model.state_dict(), model_name)
				logger.info("Saved Model, Epoch:%s, f: %.4f" % (idx, current_score))
				best_dev = current_score
				last_improved = idx

			speed, acc, p, r, f, _, _ = evalute_model(data, model, 'test', self.configs, alphabet, self.encoder_type)
			test_finish = time.time()
			test_cost = test_finish - dev_finish
			logger.info("Epoch: %s, Loss: %s, Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
				idx, total_loss, test_cost, speed, acc, p, r, f))

			if idx - last_improved > self.configs['require_improvement']:
				logger.info('No optimization for %s epoch, auto-stopping' % self.configs['require_improvement'])
				writer.close()
				break
		writer.close()

	@staticmethod
	def lr_decay(optimzer, epoch, decay_rate, init_lr):
		lr = init_lr / (1 + decay_rate * epoch)
		logging.info("Learning rate is set as: %s", lr)
		for param_group in optimzer.param_groups:
			param_group['lr'] = lr
		return optimzer

	@classmethod
	def read_configs(cls):
		with open(yml_path, 'r') as rf:
			configs = yaml.load(rf, Loader=yaml.FullLoader)
		# 读取设备基本属性
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		gpu = True if device.type == 'cuda' else False
		map_location = 'cpu' if gpu is False else None
		configs.update({'device': device, 'gpu': gpu, 'map_location': map_location})
		# 读取model_num对应的模型超参数
		model_num = configs['model_num']
		for k, v in configs['model'][model_num].items():
			configs[k] = v
		del configs['model']
		return cls(configs)


if __name__ == '__main__':
	run = Run.read_configs()
	alphabet = Alphabet.build_alphabet(data_file, run.trees.lexi_trees)
	data = Data.read_instance(alphabet, data_file, run.trees.lexi_trees, read_type='char')
	run.train(data, alphabet)
