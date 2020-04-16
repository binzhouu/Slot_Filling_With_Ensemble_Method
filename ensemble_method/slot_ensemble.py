# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2020-04-16
"""
import torch.nn as nn
import torch
import numpy as np
from datetime import datetime
from models.slot_filling.inference.inference_by_bilstm import SlotModelV1
from models.slot_filling.inference.inference_by_mul_bilstm import SlotModelV4
from models.slot_filling.utils.functions import predict_recover_label


class SlotEnsemble(object):
	def __init__(self):
		self.model_v1 = SlotModelV1()
		self.model_v4 = SlotModelV4()

	def run_ensemble(self, text, intent, session_keep, previous_intent):
		chars = list(text)
		avg_emission, avg_transition, mask = self.get_features(text, intent, session_keep, previous_intent)
		ensemble_model = self.model_v1.model.crf
		ensemble_model.transitions = nn.Parameter(avg_transition)
		scores, tag_seq = ensemble_model._viterbi_decode(avg_emission, mask)
		# 推理时仅有1个tensor,因此取0
		batch_charrecover = torch.tensor([0])
		pred_result = predict_recover_label(tag_seq, mask, self.model_v1.alphabet, batch_charrecover)
		pred_result = list(np.array(pred_result).reshape(len(chars), ))
		result = self.model_v1.slot_concat(chars, pred_result)
		return result

	# 计算emission_score和transition_score
	def get_features(self, text, intent, session_keep, previous_intent):
		v1_emission, v1_mask = self.model_v1.get_emission_score(text, intent, session_keep, previous_intent)
		v4_emission, _ = self.model_v4.get_emission_score(text, intent, session_keep, previous_intent)
		avg_emission = (v1_emission + v4_emission)/2

		v1_transition = self.model_v1.model.crf.transitions
		v4_transition = self.model_v4.model.crf.transitions
		avg_transition = (v1_transition + v4_transition)/2
		return avg_emission, avg_transition, v1_mask


if __name__ == '__main__':
	slot_ensemble = SlotEnsemble()
	texts = [
		'帮我关浴霸的UV杀菌', '呼叫客房清洁服务', '查一下厨房的消毒柜干燥模式打开了吗', '开启儿童房传感器UV杀菌模式', '5天后给我预约一下维修服务', '请让酒店给我送555双订书器']
	intents = [
		'close_UV', 'room_clean', 'query_dry', 'open_UV', 'room_maintain', 'room_supplement']
	for t, i in zip(texts, intents):
		start = datetime.now()
		res = slot_ensemble.run_ensemble(t, i, False, None)
		print('text: %s, intent: %s res: %s, time costs: %s' % (t, i, res, (datetime.now() - start).total_seconds()))
