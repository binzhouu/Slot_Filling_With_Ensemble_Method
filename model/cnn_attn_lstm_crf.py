# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-03-19
"""
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from models.slot_filling.model.crf import CRF
from models.slot_filling.model.self_attn import ScaledDotProductAttention


class CnnAttnLstmCRF(nn.Module):
	def __init__(self, data, configs):
		super(CnnAttnLstmCRF, self).__init__()
		if configs['random_embedding']:
			self.char_embeddings = nn.Embedding(data.char_alphabet_size, configs['char_emb_dim'])
			self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
				data.char_alphabet_size, configs['char_emb_dim'])))
			self.char_drop = nn.Dropout(configs['dropout'])

			self.word_embeddings = nn.Embedding(data.word_alphabet_size, configs['word_emb_dim'])
			self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
				data.word_alphabet_size, configs['word_emb_dim'])))
			# self.word_drop = nn.Dropout(configs['dropout'])

			self.lexi_embeddings = nn.Embedding(data.lexicon_alphabet_size, configs['lexi_emb_dim'])
			self.lexi_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(
				data.lexicon_alphabet_size, configs['lexi_emb_dim'])))
		else:
			pass
		# self.word_feat_embeddings = nn.Embedding()
		self.intent_embeddings = nn.Embedding(data.feat_alphabet_size, configs['intent_emb_dim'])
		self.word_drop = nn.Dropout(configs['dropout'])
		self.char_cnn = nn.Conv1d(
			in_channels=configs['char_emb_dim'], out_channels=configs['cnn_hidden_dim'], kernel_size=3, padding=1)

		self.lstm = nn.LSTM(
			configs['cnn_hidden_dim'] + configs['intent_emb_dim'], configs['lstm_hidden_dim']//2,
			num_layers=configs['num_layers'], batch_first=configs['batch_first'], bidirectional=configs['bidirectional'])
		self.drop_lstm = nn.Dropout(configs['dropout'])
		self.hidden2tag = nn.Linear(configs['lstm_hidden_dim'], data.label_alphabet_size + 2)
		self.crf = CRF(data.label_alphabet_size, configs['gpu'])

		temperature = np.power(configs['char_emb_dim'], 0.5)
		self.attention = ScaledDotProductAttention(temperature)

	def forward(self, batch_word, batch_intents, batch_wordlen, batch_char, batch_charlen, mask, batch_lexi, batch_label=None):
		char_embeds = self.char_drop(self.char_embeddings(batch_char)).transpose(1, 2)
		char_cnn_out = self.char_cnn(char_embeds).transpose(1, 2)
		# char_cnn_out = torch.max_pool1d(char_cnn_out, kernel_size=char_cnn_out.size(2)).view(char_batch_size, -1)
		intent_embeds = self.intent_embeddings(batch_intents)
		char_intent_embeds = torch.repeat_interleave(intent_embeds, batch_char.size(1), dim=1)

		char_features = torch.cat([char_cnn_out, char_intent_embeds], 2)  # (b, 14, 300)

		word_embeds = self.word_drop(self.word_embeddings(batch_word))  # (b, 9, 280)
		lexi_embeds = self.lexi_embeddings(batch_lexi)  # (b, 9, 20)
		word_intent_embeds = torch.repeat_interleave(intent_embeds, batch_word.size(1), dim=1)  # (b, 9, 100)
		word_features = torch.cat([word_embeds, lexi_embeds, word_intent_embeds], 2)  # (b, 9, 280+20+100)

		q = char_features  # (14, 400)
		k = word_features  # (9, 400)
		v = word_features  # (9, 400)
		attn_output, _ = self.attention(q, k, v)  # (14, 300)

		packed_words = pack_padded_sequence(attn_output, batch_charlen.cpu().numpy(), batch_first=True)
		hidden = None
		lstm_out, hidden = self.lstm(packed_words, hidden)
		lstm_out, _ = pad_packed_sequence(lstm_out)
		lstm_out = self.drop_lstm(lstm_out.transpose(1, 0))  # (b, 14, 200)

		outputs = self.hidden2tag(lstm_out)  # (b, 14, 30)

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
