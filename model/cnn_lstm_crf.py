# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-11-11
"""

import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from models.slot_filling.model.crf import CRF
import os
import numpy as np

head_path, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))


class CnnLstmConfig(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu = True if device.type == 'cuda' else False
    map_location = 'cpu' if device.type == 'cpu' else None

    batch_size = 10
    lr = 2e-2
    lr_decay = 0.05
    l2 = 1e-8
    momentum = 0
    epoch = 1000
    dropout = 0.5
    # max_sent_length = 250  # word级别的句子最大长度

    word_emb_dim = 300
    char_emb_dim = 300
    word_hidden_dim = 200
    char_hidden_dim = 50
    feature_emb_dim = 100  # 等同于word_emb_dim + char_hidden_dim

    model_path = 'torch_model/cnn_lstm_crf/best_model'

    require_improvement = 30  # 10个epoch

# config = CnnLstmConfig()


class CnnLstmCrf(nn.Module):
    def __init__(self, data, config):
        super(CnnLstmCrf, self).__init__()
        self.char_embeddings = nn.Embedding(data.char_alphabet_size, config.char_emb_dim)
        self.char_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.char_alphabet_size,
                                                                                      config.char_emb_dim)))
        self.char_drop = nn.Dropout(config.dropout)
        self.char_cnn = nn.Conv1d(in_channels=config.char_emb_dim, out_channels=config.char_hidden_dim, kernel_size=3,
                                  padding=1)

        self.word_embeddings = nn.Embedding(data.word_alphabet_size, config.word_emb_dim)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet_size,
                                                                                      config.word_emb_dim)))
        self.word_drop = nn.Dropout(config.dropout)

        self.feature_embeddings = nn.Embedding(data.feat_alphabet_size, config.feature_emb_dim)

        self.lstm = nn.LSTM(config.char_hidden_dim + config.word_emb_dim + config.feature_emb_dim,
                            config.word_hidden_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(config.dropout)

        self.hidden2tag = nn.Linear(config.word_hidden_dim, data.label_alphabet_size + 2)
        self.crf = CRF(data.label_alphabet_size, config.gpu)

    def forward(self, batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover,
                mask, batch_label=None):
        char_batch_size = batch_char.size(0)
        char_embeds = self.char_drop(self.char_embeddings(batch_char)).transpose(1, 2)
        char_cnn_out = self.char_cnn(char_embeds)
        char_cnn_out = torch.max_pool1d(char_cnn_out, kernel_size=char_cnn_out.size(2))  # 在hidden的维度做最大池化
        char_cnn_out = char_cnn_out.view(char_batch_size, -1)  # shape=(词的数量, 最大词的长度)
        char_cnn_out = char_cnn_out[batch_charrecover]
        char_features = char_cnn_out.view(batch_word.size(0), batch_word.size(1), -1)

        feat_embeds = self.feature_embeddings(batch_features)  # (10, 1, 50)
        feat_embeds = torch.repeat_interleave(feat_embeds, batch_word.size(1), dim=1)

        word_embeds = self.word_embeddings(batch_word)
        word_embeds = torch.cat([word_embeds, char_features, feat_embeds], 2)
        # word_embeds = word_embeds * feat_embeds  # 与intent的特征向量做点乘，映射到每一个词上
        word_represent = self.word_drop(word_embeds)

        packed_words = pack_padded_sequence(word_represent, batch_wordlen.cpu().numpy(), batch_first=True)
        hidden = None
        lstm_out, hidden = self.lstm(packed_words, hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out)
        lstm_out = self.drop_lstm(lstm_out.transpose(1,0))

        outputs = self.hidden2tag(lstm_out)

        if batch_label is not None:
            total_loss = self.crf.neg_log_likelihood_loss(outputs, mask, batch_label)
            scores, tag_seq = self.crf._viterbi_decode(outputs, mask)
            return total_loss, tag_seq
        else:
            scores, tag_seq = self.crf._viterbi_decode(outputs, mask)
            return tag_seq

    @staticmethod
    def random_embedding( vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb
