# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-11-11
"""
import torch
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():  # 如果字符串为数字组成，则为True
            # print('char:', char)
            new_word += '0'
        # print('new_word:', new_word)
        else:
            new_word += char
    return new_word


def generate_char(char, word, alphabet):
    start = 0
    char_list, char_id_list = [], []
    for w in word:
        end = start + len(w)
        chars = char[start:end]
        char_ids = [alphabet.get_index(c, 'char') for c in chars]
        char_list.append(chars)
        char_id_list.append(char_ids)
        start = end
    return char_list, char_id_list


def store_list_element(lists):
    result = []
    for one_sentence_lists in lists:
        one_sentence_res = []
        for ids in one_sentence_lists:
            for id in ids:
                one_sentence_res.append(id)
        result.append(one_sentence_res)
    return result


# 字符模型用的方法：
def batch_char_sequence_labeling(input_batch_list, gpu, if_train=True, if_label=True):
    batch_size = len(input_batch_list)
    if if_label:
        chars = [sent[0] for sent in input_batch_list]
        features = [sent[2] for sent in input_batch_list]
        labels = [sent[3] for sent in input_batch_list]
        chars = store_list_element(chars)

        char_seq_lengths = torch.tensor(list(map(len, chars)), dtype=torch.long)
        max_seq_len = char_seq_lengths.max().item()
        # feat的长度和word的长度不相等
        feat_seq_lengths = torch.tensor(list(map(len, features)), dtype=torch.long)
        max_feat_len = feat_seq_lengths.max().item()
        # padding前准备
        char_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        feature_seq_tensor = torch.zeros((batch_size, max_feat_len), requires_grad=if_train).long()

        mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
        # padding，先把tensor全部zeros，再根据len填充
        for idx, (seq, label, seq_len) in enumerate(zip(chars, labels, char_seq_lengths)):
            # print(idx, seq, label, seq_len)
            seqlen = seq_len.item()
            char_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
            try:
                label_seq_tensor[idx, :seqlen] = torch.tensor(label, dtype=torch.long)
            except RuntimeError:
                print(seq, label, seq_len)
                break
            mask[idx, :seqlen] = torch.tensor([1] * seqlen, dtype=torch.long)
            feature_seq_tensor[idx, :seqlen] = torch.tensor(features[idx], dtype=torch.long)

        # 将样本按句子的长度(chars的len降序)
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]
        feature_seq_tensor = feature_seq_tensor[char_perm_idx]
        label_seq_tensor = label_seq_tensor[char_perm_idx]
        mask = mask[char_perm_idx]

        # 还原顺序的index:
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)

        if gpu:
            char_seq_tensor = char_seq_tensor.cuda()
            feature_seq_tensor = feature_seq_tensor.cuda()
            char_seq_lengths = char_seq_lengths.cuda()
            char_seq_recover = char_seq_recover.cuda()
            label_seq_tensor = label_seq_tensor.cuda()
            mask = mask.cuda()
    else:
        chars = [sent[0] for sent in input_batch_list]
        features = [sent[-1] for sent in input_batch_list]

        char_seq_lengths = torch.tensor(list(map(len, chars)), dtype=torch.long)
        max_seq_len = char_seq_lengths.max().item()

        feat_seq_lengths = torch.tensor(list(map(len, features)), dtype=torch.long)
        max_feat_len = feat_seq_lengths.max().item()

        char_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        feature_seq_tensor = torch.zeros((batch_size, max_feat_len), requires_grad=if_train).long()

        mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()

        for idx, (seq, seq_len) in enumerate(zip(chars, char_seq_lengths)):
            seqlen = seq_len.item()  # 未padding的长度
            char_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
            mask[idx, :seqlen] = torch.tensor([1] * seqlen, dtype=torch.long)
            feature_seq_tensor[idx, :seqlen] = torch.tensor(features[idx], dtype=torch.long)

        char_seq_recover = torch.tensor([0])
        label_seq_tensor = None

        if gpu:
            char_seq_tensor = char_seq_tensor.cuda()
            feature_seq_tensor = feature_seq_tensor.cuda()
            char_seq_lengths = char_seq_lengths.cuda()
            char_seq_recover = char_seq_recover.cuda()
            mask = mask.cuda()

    return char_seq_tensor, feature_seq_tensor, char_seq_lengths, char_seq_recover, mask, label_seq_tensor


# 基于字符+词特征的方法：
def batch_char_sequence_labeling_with_word(input_batch_list, gpu, if_train=True, if_label=True):
    """

    :param input_batch_list: [[char_id],[word_id],[intent_id],[slot_id],[lexicon_id]]
    :param gpu:
    :param if_train:
    :param if_label:
    :return:
    """
    batch_size = len(input_batch_list)
    if if_label:
        words = [sent[1] for sent in input_batch_list]
        features = [sent[2] for sent in input_batch_list]
        chars = [sent[0] for sent in input_batch_list]
        labels = [sent[-2] for sent in input_batch_list]
        lexicons = [sent[-1] for sent in input_batch_list]

        chars = store_list_element(chars)
        # 字符:
        char_seq_lengths = torch.tensor(list(map(len, chars)), dtype=torch.long)
        max_seq_len = char_seq_lengths.max().item()

        feat_seq_lengths = torch.tensor(list(map(len, features)), dtype=torch.long)
        max_feat_len = feat_seq_lengths.max().item()

        char_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        feature_seq_tensor = torch.zeros((batch_size, max_feat_len), requires_grad=if_train).long()

        mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
        # label和label_mask padding
        for idx, (seq, label, seq_len) in enumerate(zip(chars, labels, char_seq_lengths)):
            seqlen = seq_len.item()
            char_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
            label_seq_tensor[idx, :seqlen] = torch.tensor(label, dtype=torch.long)
            mask[idx, :seqlen] = torch.tensor([1] * seqlen, dtype=torch.long)
            feature_seq_tensor[idx, :seqlen] = torch.tensor(features[idx], dtype=torch.long)
        # 排序
        char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
        char_seq_tensor = char_seq_tensor[char_perm_idx]
        feature_seq_tensor = feature_seq_tensor[char_perm_idx]
        label_seq_tensor = label_seq_tensor[char_perm_idx]
        # label级别的 mask
        mask = mask[char_perm_idx]
        # 词
        word_seq_lengths = torch.tensor(list(map(len, words)), dtype=torch.long)  # 一个batch中，每个句子len组成的list
        max_seq_word_len = word_seq_lengths.max().item()
        word_seq_tensor = torch.zeros((batch_size, max_seq_word_len), requires_grad=if_train).long()
        lexi_seq_tensor = torch.zeros((batch_size, max_seq_word_len), requires_grad=if_train).long()

        for idx, (seq, seq_lexi, seq_len) in enumerate(zip(words, lexicons, word_seq_lengths)):
            seqlen = seq_len.item()
            word_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
            lexi_seq_tensor[idx, :seqlen] = torch.tensor(seq_lexi, dtype=torch.long)
        # 词的特征按字符长度的顺序降序
        word_seq_tensor = word_seq_tensor[char_perm_idx]
        lexi_seq_tensor = lexi_seq_tensor[char_perm_idx]
        word_seq_lengths = word_seq_lengths[char_perm_idx]

        # 还原顺序
        _, char_seq_recover = char_perm_idx.sort(0, descending=False)
        _, word_seq_recover = char_perm_idx.sort(0, descending=False)

        if gpu:
            word_seq_tensor = word_seq_tensor.cuda()
            feature_seq_tensor = feature_seq_tensor.cuda()
            word_seq_lengths = word_seq_lengths.cuda()
            word_seq_recover = word_seq_recover.cuda()
            char_seq_tensor = char_seq_tensor.cuda()
            char_seq_recover = char_seq_recover.cuda()
            label_seq_tensor = label_seq_tensor.cuda()
            mask = mask.cuda()
            lexi_seq_tensor = lexi_seq_tensor.cuda()
    else:
        words = [sent[1] for sent in input_batch_list]
        features = [sent[2] for sent in input_batch_list]
        chars = [sent[0] for sent in input_batch_list]
        lexicons = [sent[-1] for sent in input_batch_list]
        # 字符:
        char_seq_lengths = torch.tensor(list(map(len, chars)), dtype=torch.long)
        max_seq_len = char_seq_lengths.max().item()

        feat_seq_lengths = torch.tensor(list(map(len, features)), dtype=torch.long)
        max_feat_len = feat_seq_lengths.max().item()

        char_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
        feature_seq_tensor = torch.zeros((batch_size, max_feat_len), requires_grad=if_train).long()

        mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
        # label和label_mask padding
        for idx, (seq, seq_len) in enumerate(zip(chars, char_seq_lengths)):
            seqlen = seq_len.item()
            char_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
            mask[idx, :seqlen] = torch.tensor([1] * seqlen, dtype=torch.long)
            feature_seq_tensor[idx, :seqlen] = torch.tensor(features[idx], dtype=torch.long)
        # 词
        word_seq_lengths = torch.tensor(list(map(len, words)), dtype=torch.long)  # 一个batch中，每个句子len组成的list
        max_seq_word_len = word_seq_lengths.max().item()
        word_seq_tensor = torch.zeros((batch_size, max_seq_word_len), requires_grad=if_train).long()
        lexi_seq_tensor = torch.zeros((batch_size, max_seq_word_len), requires_grad=if_train).long()

        for idx, (seq, seq_lexi, seq_len) in enumerate(zip(words, lexicons, word_seq_lengths)):
            seqlen = seq_len.item()
            word_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
            lexi_seq_tensor[idx, :seqlen] = torch.tensor(seq_lexi, dtype=torch.long)
        # 还原顺序
        word_seq_recover = torch.tensor([0])
        char_seq_recover = torch.tensor([0])
        label_seq_tensor = None

        if gpu:
            word_seq_tensor = word_seq_tensor.cuda()
            feature_seq_tensor = feature_seq_tensor.cuda()
            word_seq_lengths = word_seq_lengths.cuda()
            word_seq_recover = word_seq_recover.cuda()
            char_seq_tensor = char_seq_tensor.cuda()
            char_seq_recover = char_seq_recover.cuda()
            mask = mask.cuda()
            lexi_seq_tensor = lexi_seq_tensor.cuda()

    return word_seq_tensor, feature_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, \
        char_seq_lengths, char_seq_recover, mask, label_seq_tensor, lexi_seq_tensor


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    batch_size = len(input_batch_list)
    words = [sent[1] for sent in input_batch_list]
    features = [sent[2] for sent in input_batch_list]
    chars = [sent[0] for sent in input_batch_list]
    labels = [sent[-1] for sent in input_batch_list]

    word_seq_lengths = torch.tensor(list(map(len, words)), dtype=torch.long)  # 一个batch中，每个句子len组成的list
    max_seq_len = word_seq_lengths.max().item()  # 取一个batch中，最大长度
    # feat的长度和word的长度不相等
    feat_seq_lengths = torch.tensor(list(map(len, features)), dtype=torch.long)
    max_feat_len = feat_seq_lengths.max().item()
    # padding前准备
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()  # 取batch中最大长度，作为padding
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    feature_seq_tensor = torch.zeros((batch_size, max_feat_len), requires_grad=if_train).long()  # feat的长度是intent的数量

    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
    # padding，先把tensor全部zeros，再根据len填充
    for idx, (seq, label, seq_len) in enumerate(zip(words, labels, word_seq_lengths)):
        # print(idx, seq, label, seq_len)
        seqlen = seq_len.item()
        word_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
        label_seq_tensor[idx, :seqlen] = torch.tensor(label, dtype=torch.long)
        mask[idx, :seqlen] = torch.tensor([1] * seqlen, dtype=torch.long)  # words中非padding的部分mask
        feature_seq_tensor[idx, :seqlen] = torch.tensor(features[idx], dtype=torch.long)

    # 将样本按句子的长度(words的len降序)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]  # 按降序重排batch中的句子
    feature_seq_tensor = feature_seq_tensor[word_perm_idx]  # intent根据word_perm_idx排序，没问题
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    # chars补0到最大words的句子长度 （"len(chars[idx])"和词的长度是一样的)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]  # 每个word的char数量 统计
    max_word_len = max(map(max, length_list))  # 一个batch中，所有词的最大char长度
    # padding前准备：char_seq_tensor是padding后的形状
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.tensor(length_list, dtype=torch.long)  # batch中，每个句子，每个word的char数量
    # 开始char的padding
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.tensor(word, dtype=torch.long)
    char_seq_tensor = char_seq_tensor[word_perm_idx]  # 按词的长度先排序
    char_seq_tensor = char_seq_tensor.view(batch_size * max_seq_len, -1)  # 变换模型中char_inputs的形状
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )  # 同样，降序

    # 按word的字数长度(chars的len降序)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)  # 字级别再降序
    char_seq_tensor = char_seq_tensor[char_perm_idx]  # 字级别的输入再降序

    # 还原顺序的index：
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)  # char_seq_tensor还原的index
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)  # word_seq_tensor还原的index

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        feature_seq_tensor = feature_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()

    return word_seq_tensor, feature_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, \
        char_seq_recover, mask, label_seq_tensor


def predict_batchify_sequence_labeling(input_batch_list, gpu=False, if_train=False):
    batch_size = len(input_batch_list)
    words = [sent[1] for sent in input_batch_list]
    features = [sent[2] for sent in input_batch_list]
    chars = [sent[0] for sent in input_batch_list]
    # labels = [sent[-1] for sent in input_batch_list]

    word_seq_lengths = torch.tensor(list(map(len, words)), dtype=torch.long)  # 一个batch中，每个句子len组成的list
    max_seq_len = word_seq_lengths.max().item()  # 取一个batch中，最大长度
    # feat的长度和word的长度不相等
    feat_seq_lengths = torch.tensor(list(map(len, features)), dtype=torch.long)
    max_feat_len = feat_seq_lengths.max().item()
    # padding前准备
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()  # 取batch中最大长度，作为padding
    # label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).long()
    feature_seq_tensor = torch.zeros((batch_size, max_feat_len), requires_grad=if_train).long()

    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).byte()
    # padding，先把tensor全部zeros，再根据len填充
    for idx, (seq, seq_len) in enumerate(zip(words, word_seq_lengths)):
        seqlen = seq_len.item()
        word_seq_tensor[idx, :seqlen] = torch.tensor(seq, dtype=torch.long)
        # label_seq_tensor[idx, :seqlen] = torch.tensor(label, dtype=torch.long)
        mask[idx, :seqlen] = torch.tensor([1] * seqlen, dtype=torch.long)  # words中非padding的部分mask
        feature_seq_tensor[idx, :seqlen] = torch.tensor(features[idx], dtype=torch.long)

    # 将样本按句子的长度(words的len降序)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]  # 按降序重排batch中的句子
    feature_seq_tensor = feature_seq_tensor[word_perm_idx]  # intent根据word_perm_idx排序，没问题
    # label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]

    # chars补0到最大words的句子长度 （"len(chars[idx])"和词的长度是一样的)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]  # 每个word的char数量 统计
    max_word_len = max(map(max, length_list))  # 一个batch中，所有词的最大char长度
    # padding前准备：char_seq_tensor是padding后的形状
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad=if_train).long()
    char_seq_lengths = torch.tensor(length_list, dtype=torch.long)  # batch中，每个句子，每个word的char数量

    # 开始char的padding
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.tensor(word, dtype=torch.long)
    char_seq_tensor = char_seq_tensor[word_perm_idx]  # 按词的长度先排序
    char_seq_tensor = char_seq_tensor.view(batch_size * max_seq_len, -1)  # 变换模型中char_inputs的形状
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )  # 同样，降序

    # 按word的字数长度(chars的len降序)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)  # 字级别再降序
    char_seq_tensor = char_seq_tensor[char_perm_idx]  # 字级别的输入再降序

    # 还原顺序的index：
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)  # char_seq_tensor还原的index
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)  # word_seq_tensor还原的index

    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        feature_seq_tensor = feature_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        # label_seq_tensor = label_seq_tensor.cuda()
        mask = mask.cuda()

    return word_seq_tensor, feature_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, \
           char_seq_recover, mask


def predict_check(pred_variable, gold_variable, mask_variable):
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlap = (pred == gold)
    right_token = np.sum(overlap * mask)
    total_token = mask.sum()
    return right_token, total_token


def evalute_model(data, model, name, config, alphabet, encoder_type):
    assert name in ['dev', 'test']
    if name == 'dev':
        instances = data.dev_ids
    elif name == 'test':
        instances = data.test_ids

    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []

    model.eval()
    batch_size = config['batch_size']
    # start_time = time.time()
    train_num = len(instances)

    total_batch = train_num // batch_size + 1
    total_loss = 0
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        if encoder_type == 'cnn_attn_lstm_crf':
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, \
                mask, batch_label, batch_lexi = batch_char_sequence_labeling_with_word(instance, config['gpu'], if_train=False)
            with torch.no_grad():  # 防止在验证阶段造成梯度累计
                loss, tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, mask, batch_lexi, batch_label)
                loss = loss.item()
        else:
            batch_char, batch_features, batch_charlen, batch_charrecover, mask, batch_label = \
                batch_char_sequence_labeling(instance, config['gpu'], if_train=False)
            tag_seq = model(batch_char, batch_features, batch_charlen, batch_charrecover, mask)
            loss = 0
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, alphabet, batch_charrecover)
        pred_results += pred_label
        gold_results += gold_label
        total_loss += loss
    logger.info('%s pre_results: %s' % (name, len(pred_results)))
    logger.info('%s gold_results: %s' % (name, len(gold_results)))
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results)

    return total_loss, acc, p, r, f, pred_results, pred_scores


def evalute(data, model, name, config, alphabet, nbest=None):
    assert name in ['dev', 'test']
    if name == 'dev':
        instances = data.dev_ids
    elif name == 'test':
        instances = data.test_ids

    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []

    model.eval()
    batch_size = config.batch_size
    # start_time = time.time()
    train_num = len(instances)

    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, \
            mask, batch_label = batchify_sequence_labeling_with_label(instance, config.gpu, if_train=False)

        tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, alphabet, batch_wordrecover)

        pred_results += pred_label
        gold_results += gold_label
    logger.info('%s pre_results: %s' % (name, len(pred_results)))
    logger.info('%s gold_results: %s' % (name, len(gold_results)))
    decode_time = time.time()
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results)
    return speed, acc, p, r, f, pred_results, pred_scores


def get_ner_fmeasure(golden_lists, predict_lists, label_type='BIO'):
    sent_num = len(golden_lists)  # 句子的数量
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(sent_num):
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES" or label_type == 'BIOES':
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    logger.info("label_type: %s, jiaoji: %s, true: %s, predict: %s" % (label_type, right_num, golden_num, predict_num))
    if predict_num == 0:
        precision = -1
    else:
        precision = (right_num + 0.0) / predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num + 0.0) / golden_num
    if (precision == -1) or (recall == -1) or (precision + recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2 * precision * recall / (precision + recall)

    accuracy = (right_tag + 0.0) / all_tag
    if label_type.upper().startswith("B-"):
        logger.info("golden_num= %s, predict_num= %s, right_num= %s" % (golden_num, predict_num, right_num))
    else:
        logger.info("Right token= %s, All token= %s, acc= %s" % (right_tag, all_tag, accuracy))
    return accuracy, precision, recall, f_measure


def get_ner_BIO(label_list):
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
            else:
                tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
                index_tag = current_label.replace(begin_label, "", 1)
        elif inside_label in current_label:
            if current_label.replace(inside_label, "", 1) == index_tag:
                whole_tag = whole_tag
            else:
                if (whole_tag != '') & (index_tag != ''):
                    tag_list.append(whole_tag + ',' + str(i - 1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '') & (index_tag != ''):
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = ''
            index_tag = ''
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(begin_label, "", 1) + '[' + str(i)
            index_tag = current_label.replace(begin_label, "", 1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i - 1))
            whole_tag = current_label.replace(single_label, "", 1) + '[' + str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '') & (index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i] + ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover,
                  sentence_classification=False):
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy], key='label') for idy in range(seq_len) if
                    mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy], key='label') for idy in range(seq_len) if
                    mask[idx][idy] != 0]
            assert (len(pred) == len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
    return pred_label, gold_label


def predict_recover_label(pred_variable, mask_variable, label_alphabet, word_recover,
                          sentence_classification=False):
    pred_variable = pred_variable[word_recover]
    # gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        # gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        # gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = pred_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        # gold_tag = gold_variable.cpu().data.numpy()
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy], key='label') for idy in range(seq_len) if
                    mask[idx][idy] != 0]
            # gold = [label_alphabet.get_instance(gold_tag[idx][idy], key='label') for idy in range(seq_len) if
            #         mask[idx][idy] != 0]
            # assert (len(pred) == len(gold))
            pred_label.append(pred)
            # gold_label.append(gold)
    return pred_label
