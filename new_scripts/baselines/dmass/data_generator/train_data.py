import copy as cp
import random as rd

import numpy as np
from nltk import word_tokenize
from copy import deepcopy
import time
import random as rd

from data_generator.vocab import Vocab
from data_generator.rule import Rule
from util import constant


class TrainData:
    def __init__(self, model_config):
        self.model_config = model_config

        vocab_simple_path = self.model_config.vocab_simple
        vocab_complex_path = self.model_config.vocab_complex
        vocab_all_path = self.model_config.vocab_all
        if self.model_config.subword_vocab_size > 0:
            vocab_simple_path = self.model_config.subword_vocab_simple
            vocab_complex_path = self.model_config.subword_vocab_complex
            vocab_all_path = self.model_config.subword_vocab_all

        data_simple_path = self.model_config.train_dataset_simple
        data_complex_path = self.model_config.train_dataset_complex

        if (self.model_config.tie_embedding == 'none' or
                    self.model_config.tie_embedding == 'dec_out'):
            self.vocab_simple = Vocab(model_config, vocab_simple_path)
            self.vocab_complex = Vocab(model_config, vocab_complex_path)
        elif (self.model_config.tie_embedding == 'all' or
                    self.model_config.tie_embedding == 'enc_dec'):
            self.vocab_simple = Vocab(model_config, vocab_all_path)
            self.vocab_complex = Vocab(model_config, vocab_all_path)

        self.size = self.get_size(data_complex_path)
        if self.model_config.use_dataset2:
            self.size2 = self.get_size(self.model_config.train_dataset_complex2)
        # Populate basic complex simple pairs
        if not self.model_config.it_train:
            self.data = self.populate_data(data_complex_path, data_simple_path,
                                           self.vocab_complex, self.vocab_simple, True)
        else:
            self.data_it = self.get_data_sample_it(data_simple_path, data_complex_path)

        print('Use Train Dataset: \n Simple\t %s. \n Complex\t %s. \n Size\t %d.'
              % (data_simple_path, data_complex_path, self.size))

        if 'rule' in self.model_config.memory or 'rule' in self.model_config.rl_configs:
            self.vocab_rule = Rule(model_config, self.model_config.vocab_rules)
            self.rules_target, self.rules_align = self.populate_rules(
                self.model_config.train_dataset_complex_ppdb, self.vocab_rule)
            print(len(self.rules_align))
            print(self.rules_align[0])
            print(self.rules_align[94206])
            print(self.size)
            print(len(self.rules_target))
            print(self.rules_target[0])
            print(self.rules_target[94206])
            assert len(self.rules_align) == self.size
            assert len(self.rules_target) == self.size
            print('Populate Rule with size:%s' % self.vocab_rule.get_rule_size())
            # if self.model_config.use_dataset2:
            #     self.rules2 = self.populate_rules(
            #         self.model_config.train_dataset_complex_ppdb2, self.vocab_rule)
            #     assert len(self.rules2) == self.size2

    def process_line(self, line, vocab, max_len, need_raw=False):
        if self.model_config.tokenizer == 'split':
            words = line.split()
        elif self.model_config.tokenizer == 'nltk':
            words = word_tokenize(line)
        else:
            raise Exception('Unknown tokenizer.')

        words = [Vocab.process_word(word, self.model_config)
                 for word in words]
        if need_raw:
            words_raw = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
        else:
            words_raw = None

        if self.model_config.subword_vocab_size > 0:
            words = [constant.SYMBOL_START] + words + [constant.SYMBOL_END]
            words = vocab.encode(' '.join(words))
        else:
            words = [vocab.encode(word) for word in words]
            words = ([self.vocab_simple.encode(constant.SYMBOL_START)] + words +
                     [self.vocab_simple.encode(constant.SYMBOL_END)])

        if self.model_config.subword_vocab_size > 0:
            pad_id = vocab.encode(constant.SYMBOL_PAD)
        else:
            pad_id = [vocab.encode(constant.SYMBOL_PAD)]

        if len(words) < max_len:
            num_pad = max_len - len(words)
            words.extend(num_pad * pad_id)
        else:
            words = words[:max_len]

        return words, words_raw

    def get_size(self, data_complex_path):
        return len(open(data_complex_path, encoding='utf-8').readlines())

    def get_data_sample_it(self, data_simple_path, data_complex_path):
        f_simple = open(data_simple_path, encoding='utf-8')
        f_complex = open(data_complex_path, encoding='utf-8')
        # if self.model_config.use_dataset2:
        #     f_simple2 = open(self.model_config.train_dataset_simple2, encoding='utf-8')
        #     f_complex2 = open(self.model_config.train_dataset_complex2, encoding='utf-8')
        #     j = 0
        i = 0
        while True:
            if i >= self.size:
                f_simple = open(data_simple_path, encoding='utf-8')
                f_complex = open(data_complex_path, encoding='utf-8')
                i = 0
            line_complex = f_complex.readline()
            line_simple = f_simple.readline()
            if rd.random() < 0.5 or i >= self.size:
                i += 1
                continue

            words_complex, words_raw_comp = self.process_line(
                line_complex, self.vocab_complex, self.model_config.max_complex_sentence, True)
            words_simple, words_raw_simp = self.process_line(
                line_simple, self.vocab_simple, self.model_config.max_simple_sentence, True)

            supplement = {}
            if 'rule' in self.model_config.memory:
                supplement['rules_target'] = self.rules_target[i]
                supplement['rules_align'] = self.rules_align[i]

            obj = {}
            obj['words_comp'] = words_complex
            obj['words_simp'] = words_simple
            obj['words_raw_comp'] = words_raw_comp
            obj['words_raw_simp'] = words_raw_simp

            yield i, obj, supplement

            i += 1


            # if self.model_config.use_dataset2:
            #     if j == self.size2:
            #         f_simple2 = open(self.model_config.train_dataset_simple2, encoding='utf-8')
            #         f_complex2 = open(self.model_config.train_dataset_complex2, encoding='utf-8')
            #         j = 0
            #     line_complex2 = f_complex2.readline()
            #     line_simple2 = f_simple2.readline()
            #     words_complex2, _ = self.process_line(line_complex2, self.vocab_complex)
            #     words_simple2, _ = self.process_line(line_simple2, self.vocab_simple)
            #
            #     supplement2 = {}
            #     if self.model_config.memory == 'rule':
            #         supplement2['mem'] = self.rules2[j]
            #
            #     yield j, words_simple2, words_complex2, cp.deepcopy([1.0] * len(words_simple2)), cp.deepcopy([1.0] * len(words_complex2)), supplement2
            #     j += 1

    def populate_rules(self, rule_path, vocab_rule):
        data_target, data_align = [], []

        i = 0
        for line in open(rule_path, encoding='utf-8'):
            i += 1
            cur_rules = line.split('\t')
            tmp, tmp_align = [], []
            for cur_rule in cur_rules:
                rule_id, rule_origins, rule_targets = vocab_rule.encode(cur_rule)
                if rule_targets is not None and rule_origins is not None:
                    tmp.append((rule_id, [self.vocab_simple.encode(rule_target) for rule_target in rule_targets]))

                    if len(rule_origins) == 1 and len(rule_targets) == 1:
                        tmp_align.append(
                            (self.vocab_complex.encode(rule_origins[0]),
                             self.vocab_simple.encode(rule_targets[0])))
            data_target.append(tmp)
            data_align.append(tmp_align)

        ## ADDED CODE: THIS MAY BE WRONG!!!!!
        data_target.append([])
        data_align.append([])

        return data_target, data_align

    def populate_data(self, data_path_comp, data_path_simp, vocab_comp, vocab_simp, need_raw=False):
        # Populate data into memory
        data = []
        # max_len = -1
        # from collections import Counter
        # len_report = Counter()
        lines_comp = open(data_path_comp, encoding='utf-8').readlines()
        lines_simp = open(data_path_simp, encoding='utf-8').readlines()
        assert len(lines_comp) == len(lines_simp)
        for line_id in range(len(lines_comp)):
            obj = {}
            line_comp = lines_comp[line_id]
            line_simp = lines_simp[line_id]
            words_comp, words_raw_comp = self.process_line(
                line_comp, vocab_comp, self.model_config.max_complex_sentence, need_raw)
            words_simp, words_raw_simp = self.process_line(
                line_simp, vocab_simp, self.model_config.max_simple_sentence, need_raw)
            obj['words_comp'] = words_comp
            obj['words_simp'] = words_simp
            if need_raw:
                obj['words_raw_comp'] = words_raw_comp
                obj['words_raw_simp'] = words_raw_simp

            data.append(obj)
        return data

    def get_data_sample(self):
        i = rd.sample(range(self.size), 1)[0]
        supplement = {}
        if 'rule' in self.model_config.memory:
            supplement['rules_target'] = self.rules_target[i]
            supplement['rules_align'] = self.rules_align[i]

        return i, self.data[i], supplement



