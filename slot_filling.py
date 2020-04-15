# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-11-15
"""
import os, sys
import logging
import pickle
from models.slot_filling.utils.alphabet import Alphabet
import torch
from models.slot_filling.utils.functions import predict_batchify_sequence_labeling, \
    predict_recover_label, normalize_word
from models.slot_filling.model.cnn_lstm_crf import CnnLstmCrf, CnnLstmConfig
from models.slot_filling.utils.data import Data
import ast
from constants import ROOT_PATH
import re
import numpy as np

logger = logging.getLogger(__name__)
head_path, _ = os.path.split(os.path.abspath(__file__))

# ROOT_PATH = '/'.join(os.path.abspath(__file__).split('/')[:-1])
# print(ROOT_PATH)
# sys.path.append(ROOT_PATH)


class Slots(object):
    def __init__(self):
        dset_file = os.path.join(head_path, 'data/output/alphabet.dset')
        with open(dset_file, 'rb') as rbf:
            char_alphabet = pickle.load(rbf)
            word_alphabet = pickle.load(rbf)
            feat_alphabet = pickle.load(rbf)
            label_alphabet = pickle.load(rbf)
            char_alphabet_size = len(char_alphabet) + 1
            word_alphabet_size = len(word_alphabet) + 1
            feat_alphabet_size = len(feat_alphabet) + 1
            label_alphabet_size = len(label_alphabet) + 1
        data = Data([], [], [], [], [], [], char_alphabet_size, word_alphabet_size, feat_alphabet_size,
                    label_alphabet_size)
        model_dir = os.path.join(ROOT_PATH, 'saved_models/slot_filling/bilstm_crf.model')
        self.config = CnnLstmConfig()
        self.alphabet = Alphabet(word_alphabet, char_alphabet, feat_alphabet, label_alphabet, [], [], [], [])
        self.model = CnnLstmCrf(data, self.config)
        self.model.load_state_dict(torch.load(model_dir, map_location=self.config.map_location))
        self.model.eval()
        self.model.to(self.config.device)

    def slot_filling(self, text_list, intent):
        instance_ids = []
        word_ids, char_ids, feat_ids = [], [], []
        feat_ids.append([self.alphabet.get_index(intent, 'intent')])
        for word in text_list:
            char_id = []
            word_ids.append(self.alphabet.get_index(normalize_word(word), 'word'))
            for char in word:
                char_id.append(self.alphabet.get_index(normalize_word(char), 'char'))
            char_ids.append(char_id)
            # char_ids.append([self.alphabet.get_index(char, 'char') for char in word])
        instance_ids.append([char_ids, word_ids, feat_ids])
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, \
            mask = predict_batchify_sequence_labeling(instance_ids, self.config.gpu)
        with torch.no_grad():
            tag_seq = self.model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                 batch_charrecover, mask)
        pred_result = predict_recover_label(tag_seq, mask, self.alphabet, batch_wordrecover)
        pred_result = list(np.array(pred_result).reshape(len(text_list),))
        result = self.slot_concat(text_list, pred_result)
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

    # def test_one(self):
    #     while True:
    #         text = input('text:')
    #         if text == 'q':
    #             break
    #         text = ast.literal_eval(text)
    #         intent = input('intent:')
    #         input_dict = {'text': text, 'intent': intent}
    #         res = self.slot_filling(input_dict)
    #         print('slots:', res)


if __name__ == '__main__':
    slots = Slots()
    # inputs = {'text': ['空调', '调高', '20'], 'intent': 'increment_temperature'}
    texts, intent = ['晾衣架', '调高', '20%'], 'increment_height'
    # texts, intent = ['空调', '调高', '20'], 'increment_temperature'
    # texts, intent = ["我", "想", "看", "上周", "CCTV", "3", "播放", "的", "朝闻天下"], 'query_videos'
    # texts, intent = ["打开", "三", "路", "触控", "开关", "和", "空气", "净化器"], 'turn_on'
    # texts, intent = ["打开", "卧室", "香薰机", "的", "氛围", "灯"], 'turn_on'
    # texts, intent = ["打开", "空调", "和", "空气", "净化器"], "turn_on"
    # inputs = {'text': ['空调', '调高', '20', '度'], 'intent': 'increment_temperature'}
    res = slots.slot_filling(texts, intent)
    print(res)
    # slots.test_one()
