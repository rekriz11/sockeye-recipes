# For fix slurm cannot load PYTHONPATH
import sys
# sys.path.insert(0,'/ihome/hdaqing/saz31/sanqiang/text_simplification')
# sys.path.insert(0,'/home/hed/text_simp_proj/text_simplification')
#sys.path.insert(0,'/ihome/cs2770_s2018/maz54/ts/text_simplification')
#sys.path.insert(0,'/home/zhaos5/ts/text_simplification')
sys.path.insert(0,'/home/rekriz/sockeye-recipes/new_scripts/baselines/dmass/')

from data_generator.train_data import TrainData
from model.transformer import TransformerGraph
from model.seq2seq import Seq2SeqGraph
from model.model_config import DefaultConfig, DefaultTrainConfig, list_config
from model.model_config import WikiDressLargeNewTrainDefault, WikiDressHugeNewTrainDefault,WikiDressLargeTrainDefault
from data_generator.vocab import Vocab
from util import session
from util import constant
from model.eval import eval, get_ckpt

import tensorflow as tf
import tensorflow.contrib.slim as slim
from util.arguments import get_args
from datetime import datetime
from util.sys_moniter import print_cpu_memory, print_gpu_memory
from copy import deepcopy
from os.path import exists, dirname
from os import listdir


args = get_args()


def find_best_ckpt(model_config):
    dir = dirname(model_config.logdir[:-1] + '2/')
    files = listdir(model_config.logdir[:-1] + '2/')
    max_id = -1
    for file in files:
        if file.startswith('model.ckpt-') and file.endswith('.meta'):
            sid = file.index('model.ckpt-') + len('model.ckpt-')
            eid = file.rindex('.')
            id = int(file[sid:eid])
            max_id = max(id, max_id)
    return ''.join([dir, '/model.ckpt-', str(max_id)])


def get_graph_train_data(
        data,
        objs,
        model_config):
    input_feed = {}
    for obj in objs:
        (tmp_sentence_simple, tmp_sentence_complex,
         tmp_sentence_simple_weight, tmp_attn_weight,
         tmp_idxs, tmp_sups, tmp_sentence_simple_raw, tmp_sentence_complex_raw) = [], [], [], [], [], {}, [], []

        for i in range(model_config.batch_size):
            if not model_config.it_train:
                idx, obj_data, sup = data.get_data_sample()
            else:
                idx, obj_data, sup = next(data.data_it)

            tmp_sentence_simple.append(obj_data['words_simp'])
            tmp_sentence_complex.append(obj_data['words_comp'])
            tmp_idxs.append(idx)

            if 'rule' in model_config.memory:
                if 'rule_id_input_placeholder' not in tmp_sups:
                    tmp_sups['rule_id_input_placeholder'] = []
                if 'rule_target_input_placeholder' not in tmp_sups:
                    tmp_sups['rule_target_input_placeholder'] = []
                if 'rule_pair_input_placeholder' not in tmp_sups:
                    tmp_sups['rule_pair_input_placeholder'] = []

                cur_rule_id_input_placeholder = []
                cur_rule_target_input_placeholder = []
                for rule_tuple in sup['rules_target']:
                    rule_id = rule_tuple[0]
                    rule_targets = rule_tuple[1]
                    for target in rule_targets:
                        cur_rule_id_input_placeholder.append(rule_id)
                        cur_rule_target_input_placeholder.append(target)

                if len(cur_rule_id_input_placeholder) < model_config.max_cand_rules:
                    num_pad = model_config.max_cand_rules - len(cur_rule_id_input_placeholder)
                    cur_rule_id_input_placeholder.extend(num_pad * [0])
                    cur_rule_target_input_placeholder.extend(num_pad * [data.vocab_simple.encode(constant.SYMBOL_PAD)])
                else:
                    cur_rule_id_input_placeholder = cur_rule_id_input_placeholder[:model_config.max_cand_rules]
                    cur_rule_target_input_placeholder = cur_rule_target_input_placeholder[:model_config.max_cand_rules]

                tmp_sups['rule_id_input_placeholder'].append(cur_rule_id_input_placeholder)
                tmp_sups['rule_target_input_placeholder'].append(cur_rule_target_input_placeholder)

                cur_rule_align_input_placeholder = []
                for rule_tuple in sup['rules_align']:
                    rule_tar = rule_tuple[1]
                    rule_tar = obj_data['words_simp'].index(rule_tar)
                    rule_ori = rule_tuple[0]
                    rule_ori = obj_data['words_comp'].index(rule_ori)
                    cur_rule_align_input_placeholder.append([rule_tar, rule_ori])

                if len(cur_rule_align_input_placeholder) < model_config.max_cand_rules:
                    num_pad = model_config.max_cand_rules - len(cur_rule_align_input_placeholder)
                    cur_rule_align_input_placeholder.extend(
                        num_pad * [[0, 0]])
                else:
                    cur_rule_align_input_placeholder = cur_rule_align_input_placeholder[:model_config.max_cand_rules]

                tmp_sups['rule_pair_input_placeholder'].append(cur_rule_align_input_placeholder)


        for step in range(model_config.max_simple_sentence):
            input_feed[obj['sentence_simple_input_placeholder'][step].name] = [tmp_sentence_simple[batch_idx][step]
                                                            for batch_idx in range(model_config.batch_size)]
        for step in range(model_config.max_complex_sentence):
            input_feed[obj['sentence_complex_input_placeholder'][step].name] = [tmp_sentence_complex[batch_idx][step]
                                                             for batch_idx in range(model_config.batch_size)]
        input_feed[obj['sentence_idxs'].name] = [tmp_idxs[batch_idx] for batch_idx in range(model_config.batch_size)]

        if 'rule' in model_config.memory:
            for step in range(model_config.max_cand_rules):
                input_feed[obj['rule_id_input_placeholder'][step].name] = [
                    tmp_sups['rule_id_input_placeholder'][batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]
                input_feed[obj['rule_target_input_placeholder'][step].name] = [
                    tmp_sups['rule_target_input_placeholder'][batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]
                input_feed[obj['rule_pair_input_placeholder'][step].name] = [
                    tmp_sups['rule_pair_input_placeholder'][batch_idx][step]
                    for batch_idx in range(model_config.batch_size)]

    return input_feed


def train(model_config=None):
    model_config = (DefaultConfig()
                    if model_config is None else model_config)
    data = TrainData(model_config)

    graph = None
    if model_config.framework == 'transformer':
        graph = TransformerGraph(data, True, model_config)
    elif model_config.framework == 'seq2seq':
        graph = Seq2SeqGraph(data, True, model_config)
    else:
        raise NotImplementedError('Unknown Framework.')
    graph.create_model_multigpu()

    ckpt_path = None
    if model_config.warm_start:
        if model_config.warm_start == 'recent':
            ckpt_path = find_best_ckpt(model_config)
        else:
            ckpt_path = model_config.warm_start
        var_list = slim.get_variables_to_restore()
    if ckpt_path is not None:
        # Handling missing vars by ourselves
        available_vars = {}
        reader = tf.train.NewCheckpointReader(ckpt_path)
        var_dict = {var.op.name: var for var in var_list}
        for var in var_dict:
            if 'global_step' in var:
                continue
            if 'optimization' in var:
                continue
            if reader.has_tensor(var):
                var_ckpt = reader.get_tensor(var)
                var_cur = var_dict[var]
                if any([var_cur.shape[i] != var_ckpt.shape[i] for i in range(len(var_ckpt.shape))]):
                    print('Variable %s missing due to shape.', var)
                else:
                    available_vars[var] = var_dict[var]
            else:
                print('Variable %s missing.', var)

        partial_restore_ckpt = slim.assign_from_checkpoint_fn(
            ckpt_path, available_vars,
            ignore_missing_vars=False, reshape_variables=False)

    def init_fn(session):
        # Restore ckpt either from warm start or automatically get when changing optimizer
        ckpt_path = None
        if model_config.warm_start:
            ckpt_path = model_config.warm_start

        if ckpt_path is not None:
            if model_config.use_partial_restore:
                partial_restore_ckpt(session)
            else:
                try:
                    graph.saver.restore(session, ckpt_path)
                except Exception as ex:
                    print('Fully restore failed, use partial restore instead. \n %s' % str(ex))
                    partial_restore_ckpt(session)

            print('Warm start with checkpoint %s' % ckpt_path)

    sv = tf.train.Supervisor(logdir=model_config.logdir,
                             global_step=graph.global_step,
                             saver=graph.saver,
                             init_fn=init_fn,
                             save_model_secs=model_config.save_model_secs)
    sess = sv.PrepareSession(config=session.get_session_config(model_config))
    perplexitys = []
    start_time = datetime.now()
    while True:
        input_feed = get_graph_train_data(
            data,
            graph.objs,
            model_config)

        # fetches = [graph.train_op, graph.loss, graph.global_step,
        #            graph.perplexity, graph.ops, graph.attn_dists, graph.targets, graph.cs]
        # _, loss, step, perplexity, _ops , attn_dists, targets, cs = sess.run(fetches, input_feed)
        fetches = [graph.train_op, graph.loss, graph.global_step,
                   graph.perplexity, graph.ops, graph.logits]
        _, loss, step, perplexity, _, logits = sess.run(fetches, input_feed)
        perplexitys.append(perplexity)

        if step % model_config.model_print_freq == 0:
            end_time = datetime.now()
            time_span = end_time - start_time
            start_time = end_time
            print('Perplexity:\t%f at step %d using %s.' % (perplexity, step, time_span))
            perplexitys.clear()

if __name__ == '__main__':
    config = None
    if args.mode == 'dummy':
        config = DefaultTrainConfig()
    elif args.mode == 'dressnew':
        config = WikiDressLargeNewTrainDefault()
    elif args.mode == 'wikihuge':
        config = WikiDressHugeNewTrainDefault()
    elif args.mode == 'dress':
        config = WikiDressLargeTrainDefault()
    print(list_config(config))
    train(config)
