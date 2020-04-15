# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-03-04
"""

import torch
import torch.nn as nn
import os
import yaml
import numpy as np
import pickle
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from constants import ROOT_PATH
from models.slot_filling.model.crf import CRF


class BiLstmCrf(nn.Module):
	def __init__(self, data, configs):
		super(BiLstmCrf, self).__init__()
		if configs['random_embedding']:
			self.char_embeddings = nn.Embedding(data.char_alphabet_size, configs['char_emb_dim'])
			self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
				data.char_alphabet_size, configs['char_emb_dim'])))
			self.char_drop = nn.Dropout(configs['dropout'])
		else:
			pass
		self.feature_embeddings = nn.Embedding(data.feat_alphabet_size, configs['feature_emb_dim'])
		self.input_drop = nn.Dropout(configs['dropout'])
		self.lstm = nn.LSTM(
			configs['char_emb_dim'] + configs['feature_emb_dim'], configs['hidden_dim']//2,
			num_layers=configs['num_layers'], batch_first=configs['batch_first'],
			bidirectional=configs['bidirectional'])
		self.drop_lstm = nn.Dropout(configs['dropout'])
		self.hidden2tag = nn.Linear(configs['hidden_dim'], data.label_alphabet_size + 2)
		self.crf = CRF(data.label_alphabet_size, configs['gpu'])

	def forward(self, batch_input, batch_feature, batch_len, batch_recover, mask, batch_label=None):
		batch_size = len(batch_input)
		char_embeds = self.char_drop(self.char_embeddings(batch_input))
		feat_embeds = self.feature_embeddings(batch_feature)
		feat_embeds = torch.repeat_interleave(feat_embeds, batch_input.size(1), dim=1)
		# input_embeds = char_embeds * feat_embeds

		input_embeds = torch.cat([char_embeds, feat_embeds], 2)
		input_represent = self.input_drop(input_embeds)

		packed_words = pack_padded_sequence(input_represent, batch_len.cpu().numpy(), batch_first=True)
		# 不加feat_embeds(即只用char_embedding),用下面这行即可
		# packed_words = pack_padded_sequence(char_embeds, batch_len.cpu().numpy(), batch_first=True)
		hidden = None
		lstm_out, hidden = self.lstm(packed_words, hidden)
		lstm_out, _ = pad_packed_sequence(lstm_out)
		lstm_out = self.drop_lstm(lstm_out.transpose(1, 0))

		outputs = self.hidden2tag(lstm_out)

		if batch_label is not None:
			total_loss = self.crf.neg_log_likelihood_loss(outputs, mask, batch_label)
			scores, tag_seq = self.crf._viterbi_decode(outputs, mask)
			return total_loss, tag_seq
		else:
			scores, tag_seq = self.crf._viterbi_decode(outputs, mask)
			return tag_seq

	@staticmethod
	def random_embedding(vocab_size, embedding_dim):
		pretrain_emb = np.empty([vocab_size, embedding_dim])
		scale = np.sqrt(3.0 / embedding_dim)
		for index in range(vocab_size):
			pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
		return pretrain_emb


if __name__ == '__main__':
	pass
