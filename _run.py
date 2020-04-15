# -*- coding: utf-8 -*-
"""
Author: bin zhou
Date: 2019-10-22
"""
import os
import sys
ROOT_PATH = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
sys.path.append(ROOT_PATH)
import numpy as np
import torch
import logging
from models.slot_filling.model.cnn_lstm_crf import CnnLstmCrf, CnnLstmConfig
import torch.optim as optim
from models.slot_filling.utils.data import Data
from models.slot_filling.utils.alphabet import Alphabet
import datetime
import time
from models.slot_filling.utils.functions import batchify_sequence_labeling_with_label, predict_check, \
    evalute
from tensorboardX import SummaryWriter


head_path, _ = os.path.split(os.path.abspath(__file__))
data_file = os.path.join(head_path, 'data/output/intent_corpus_v3.txt')
writer = SummaryWriter('./tensorboard_log')


def init_logging(log_filename):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s\t%(levelname)s: %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        filename=log_filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s\t%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class Run(object):
    def __init__(self):
        pass

    def train(self, data, config, alphabet):
        model = CnnLstmCrf(data, config)
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.l2)
        if config.gpu:
            model = model.cuda()
        best_dev = -10
        last_improved = 0
        logging.info('train start:%s' % datetime.datetime.now())
        for idx in range(config.epoch):
            epoch_start = time.time()
            temp_start = epoch_start
            logging.info('Epoch: %s/%s' % (idx, config.epoch))
            optimizer = self.lr_decay(optimizer, idx, config.lr_decay, config.lr)

            sample_loss = 0
            total_loss = 0
            right_token = 0
            whole_token = 0
            logging.info("first input word _list: %s, %s" % (data.train_texts[0][1], data.train_ids[0][1]))

            model.train()
            model.zero_grad()
            batch_size = config.batch_size
            # data.train_ids = data.train_ids[:-3]
            train_num = len(data.train_ids)
            total_batch = train_num // batch_size + 1
            logging.info('total_batch %s' % total_batch)

            for batch_id in range(total_batch):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                instance = data.train_ids[start:end]
                if not instance:  # 如果instance不存在，跳出至下一个epoch
                    continue

                batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, \
                    batch_charrecover, mask, batch_label = batchify_sequence_labeling_with_label(
                        instance, config.gpu, if_train=True)
                loss, tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen,
                                      batch_charrecover, mask, batch_label)
                right, whole = predict_check(tag_seq, batch_label, mask)
                right_token += right
                whole_token += whole
                sample_loss += loss.item()
                total_loss += loss.item()

                if end % (batch_size * 10) == 0:
                    temp_time = time.time()
                    temp_cost = temp_time - temp_start
                    temp_start = temp_time
                    logging.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
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
            logging.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
                end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start
            logging.info("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (
                idx, epoch_cost, train_num / epoch_cost, total_loss))
            if total_loss > 1e8 or str(total_loss) == 'nan':
                raise ValueError("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")

            writer.add_scalar('Train_loss', total_loss, idx)

            speed, acc, p, r, f, _, _ = evalute(data, model, "dev", config, alphabet)
            dev_finish = time.time()
            dev_cost = dev_finish - epoch_finish
            current_score = f
            writer.add_scalar('Dev f1_score', current_score, idx)
            logging.info("Epoch: %s, Loss: %s, Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"
                         % (idx, total_loss, dev_cost, speed, acc, p, r, f))
            if current_score > best_dev:
                model_name = config.model_path + '.model'
                torch.save(model.state_dict(), model_name)
                logging.info("Saved Model, Epoch:%s, f: %.4f" % (idx, current_score))
                best_dev = current_score
                last_improved = idx

            speed, acc, p, r, f, _, _ = evalute(data, model, 'test', config, alphabet)
            test_finish = time.time()
            test_cost = test_finish - dev_finish
            logging.info("Epoch: %s, Loss: %s, Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"
                         % (idx, total_loss, test_cost, speed, acc, p, r, f))

            if idx - last_improved > config.require_improvement:
                logging.info('No optimization for %s epoch, auto-stopping' % config.require_improvement)
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


if __name__ == '__main__':
    init_logging('cnn_attn_lstm_crf.log')
    alphabet = Alphabet.build_alphabet(data_file)
    data = Data.read_instance(alphabet, data_file)
    config = CnnLstmConfig()
    run = Run()
    run.train(data, config, alphabet)
