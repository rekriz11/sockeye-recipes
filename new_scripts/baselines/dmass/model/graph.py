from model.embedding import Embedding
from model.loss import sequence_loss
from model.metric import Metric
from model.optimizer import TransformerOptimizer
from util import constant

import tensorflow as tf
import numpy as np


class Graph():
    def __init__(self, data, is_train, model_config):
        self.model_config = model_config
        self.data = data
        self.is_train = is_train
        self.model_fn = None
        self.rand_unif_init = tf.random_uniform_initializer(-0,.08, 0.08)
        self.metric = Metric(self.model_config, self.data)

    def embedding_fn(self, inputs, embedding):
        if type(inputs) == list:
            if not inputs:
                return []
            else:
                return [tf.nn.embedding_lookup(embedding, inp) for inp in inputs]
        else:
            return tf.nn.embedding_lookup(embedding, inputs)

    def output_to_logit(self, prev_out, w, b):
        prev_logit = tf.add(tf.matmul(prev_out, tf.transpose(w)), b)
        return prev_logit

    def create_model_multigpu(self):
        # with tf.Graph().as_default():
            # with tf.device('/gpu:0'):
        losses = []
        grads = []
        ops = [tf.constant(0)]
        self.objs = []
        optim = self.get_optim()

        with tf.device('/cpu:0'):
            self.global_step = tf.get_variable(
                'global_step', initializer=tf.constant(0, dtype=tf.int64), trainable=False)

        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for gpu_id in range(self.model_config.num_gpus):
                with tf.device('/device:GPU:%d' % gpu_id):
                    with tf.name_scope('%s_%d' % ('gpu_scope', gpu_id)):
                        loss, obj = self.create_model()
                        grad = optim.compute_gradients(loss)
                        tf.get_variable_scope().reuse_variables()
                        losses.append(loss)
                        grads.append(grad)
                        if 'rule' in self.model_config.memory and self.is_train:
                            ops.append(obj['mem_contexts'])
                            ops.append(obj['mem_outputs'])
                            ops.append(obj['mem_counter'])
                        self.objs.append(obj)

        with tf.variable_scope('optimization'):
            self.loss = tf.divide(tf.add_n(losses), self.model_config.num_gpus)
            self.perplexity = tf.exp(tf.reduce_mean(self.loss))

            if self.is_train:
                avg_grad = self.average_gradients(grads)
                grads = [g for (g,v) in avg_grad]
                clipped_grads, _ = tf.clip_by_global_norm(grads, self.model_config.max_grad_norm)
                self.train_op = optim.apply_gradients(zip(clipped_grads, tf.trainable_variables()), global_step=self.global_step)
                self.increment_global_step = tf.assign_add(self.global_step, 1)

            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
            self.ops = tf.tuple(ops)

    def create_model(self):
        with tf.variable_scope('variables'):
            sentence_simple_input_placeholder = []
            for step in range(self.model_config.max_simple_sentence):
                sentence_simple_input_placeholder.append(
                    tf.zeros(self.model_config.batch_size, tf.int32, name='simple_input'))

            sentence_complex_input_placeholder = []
            for step in range(self.model_config.max_complex_sentence):
                sentence_complex_input_placeholder.append(
                    tf.zeros(self.model_config.batch_size, tf.int32, name='complex_input'))

            sentence_idxs = tf.zeros(self.model_config.batch_size, tf.int32, name='sent_idx')

            embedding = Embedding(self.data.vocab_complex, self.data.vocab_simple, self.model_config)
            emb_complex = embedding.get_complex_embedding()
            emb_simple = embedding.get_simple_embedding()

            w = embedding.get_w()
            b = embedding.get_b()

            mem_contexts, mem_outputs, mem_counter = None, None, None
            rule_id_input_placeholder, rule_target_input_placeholder = [], []
            rule_pair_input_placeholder = []
            if 'rule' in self.model_config.memory:
                with tf.device('/cpu:0'):
                    context_size = 0
                    if self.model_config.framework == 'transformer':
                        context_size = 1
                    elif self.model_config.framework == 'seq2seq':
                        context_size = 2
                    mem_contexts = tf.get_variable(
                        'mem_contexts',
                        initializer=tf.constant(0, dtype=tf.float32, shape=
                        (self.data.vocab_rule.get_rule_size(), self.model_config.dimension * context_size)),
                        trainable=False, dtype=tf.float32)
                    mem_outputs = tf.get_variable(
                        'mem_outputs',
                        initializer=tf.constant(0, dtype=tf.float32, shape=(self.data.vocab_rule.get_rule_size(), self.model_config.dimension)),
                        trainable=False, dtype=tf.float32)
                    mem_counter = tf.get_variable(
                        'mem_counter',
                        initializer=tf.constant(0, dtype=tf.int32, shape=(self.data.vocab_rule.get_rule_size(), 1)),
                        trainable=False, dtype=tf.int32)

                for step in range(self.model_config.max_cand_rules):
                    rule_id_input_placeholder.append(
                        tf.zeros(self.model_config.batch_size, tf.int32, name='rule_id_input'))

                for step in range(self.model_config.max_cand_rules):
                    rule_target_input_placeholder.append(
                        tf.zeros(self.model_config.batch_size, tf.int32, name='rule_target_input'))

                for step in range(self.model_config.max_cand_rules):
                    rule_pair_input_placeholder.append(
                        tf.zeros([self.model_config.batch_size, 2], tf.int32, name='rule_pair_input'))

        with tf.variable_scope('model'):
            output = self.model_fn(sentence_complex_input_placeholder, emb_complex,
                                   sentence_simple_input_placeholder, emb_simple,
                                   w, b, rule_id_input_placeholder, mem_contexts, mem_outputs,
                                   self.global_step)

            encoder_embs, final_outputs = None, None
            if self.model_config.replace_unk_by_emb:
                encoder_embs = tf.stack(output.encoder_embed_inputs_list, axis=1)

            if output.decoder_outputs_list is not None:
                if type(output.decoder_outputs_list) == list:
                    decoder_outputs_list = output.decoder_outputs_list
                    decoder_outputs = tf.stack(decoder_outputs_list, axis=1)
                else:
                    decoder_outputs = output.decoder_outputs_list

            if output.final_outputs_list is not None:
                if type(output.final_outputs_list) == list:
                    final_outputs_list = output.final_outputs_list
                    final_outputs = tf.stack(final_outputs_list, axis=1)
                else:
                    final_outputs = output.final_outputs_list

            attn_distr = None
            if self.model_config.replace_unk_by_attn:
                attn_distr = output.attn_distr_list

            if not self.is_train:
                # in beam search, it directly provide decoder target list
                decoder_target = tf.stack(output.decoder_target_list, axis=1)
                loss = tf.reduce_mean(output.decoder_score)
                obj = {
                    'sentence_idxs': sentence_idxs,
                    'sentence_simple_input_placeholder': sentence_simple_input_placeholder,
                    'sentence_complex_input_placeholder': sentence_complex_input_placeholder,
                    'decoder_target_list': decoder_target,
                    'final_outputs':final_outputs,
                    'encoder_embs':encoder_embs,
                    'attn_distr':attn_distr
                }
                if 'rule' in self.model_config.memory:
                    obj['rule_id_input_placeholder'] = rule_id_input_placeholder
                    obj['rule_target_input_placeholder'] = rule_target_input_placeholder
                return loss, obj
            else:
                # Memory Populate
                if 'rule' in self.model_config.memory:
                    # Update Memory through python injection
                    def update_memory(
                            mem_contexts_tmp, mem_outputs_tmp, mem_counter_tmp,
                            decoder_targets, decoder_outputs, contexts,
                            rule_target_input_placeholder, rule_id_input_placeholder,
                            global_step, emb_simple, encoder_outputs):
                        if global_step <= self.model_config.memory_prepare_step:
                            return mem_contexts_tmp, mem_outputs_tmp, mem_counter_tmp

                        batch_size = np.shape(rule_target_input_placeholder)[0]
                        max_rules = np.shape(rule_target_input_placeholder)[1]
                        for batch_id in range(batch_size):
                            cur_decoder_targets = decoder_targets[batch_id, :]
                            cur_decoder_outputs = decoder_outputs[batch_id, :]
                            cur_contexts = contexts[batch_id, :]
                            cur_rule_target_input_placeholder = rule_target_input_placeholder[batch_id, :]
                            cur_rule_id_input_placeholder = rule_id_input_placeholder[batch_id, :]

                            rule_mapper = {}
                            for step in range(max_rules):
                                rule_id = cur_rule_id_input_placeholder[step]
                                if rule_id != 0:
                                    decoder_target = cur_rule_target_input_placeholder[step]
                                    if rule_id not in rule_mapper:
                                        rule_mapper[rule_id] = []
                                    rule_mapper[rule_id].append(decoder_target)

                            for rule_id in rule_mapper:
                                rule_targets = rule_mapper[rule_id]
                                decoder_target_orders = np.where(cur_decoder_targets == rule_targets[0])[0]
                                for decoder_target_order in decoder_target_orders:
                                    if len(rule_targets) > 1:
                                        if decoder_target_order+1 >= len(cur_decoder_targets) or rule_targets[1] != cur_decoder_targets[decoder_target_order+1]:
                                            continue
                                    if len(rule_targets) > 2:
                                        if decoder_target_order+2 >= len(cur_decoder_targets) or rule_targets[2] != cur_decoder_targets[decoder_target_order+2]:
                                            continue
                                    cur_context, cur_outputs = None, None
                                    for step, _ in enumerate(rule_targets):
                                        if step == 0:
                                            cur_context = cur_contexts[decoder_target_order, :]
                                            cur_outputs = cur_decoder_outputs[decoder_target_order, :]
                                        else:
                                            cur_context += cur_contexts[step+decoder_target_order, :]
                                            cur_outputs += cur_decoder_outputs[step+decoder_target_order, :]
                                    cur_context /= len(rule_targets)
                                    cur_outputs /= len(rule_targets)
                                    if mem_counter_tmp[rule_id, 0] == 0:
                                        mem_contexts_tmp[rule_id, :] = cur_context
                                        mem_outputs_tmp[rule_id, :] = cur_outputs
                                    else:
                                        mem_contexts_tmp[rule_id, :] = (cur_context + mem_contexts_tmp[rule_id, :]) / 2
                                        mem_outputs_tmp[rule_id, :] = (cur_outputs + mem_outputs_tmp[rule_id, :]) / 2
                                    mem_counter_tmp[rule_id, 0] += 1

                        return mem_contexts_tmp, mem_outputs_tmp, mem_counter_tmp

                    mem_output_input = None
                    if 'mofinal' in self.model_config.memory_config:
                        mem_output_input = final_outputs
                    # elif 'modecode' in self.model_config.memory_config:
                    #     mem_output_input = decoder_outputs
                    # elif 'moemb' in self.model_config.memory_config:
                    #     mem_output_input = tf.stack(
                    #         self.embedding_fn(sentence_simple_input_placeholder, emb_simple),
                    #         axis=1)

                    mem_contexts, mem_outputs, mem_counter = tf.py_func(update_memory,
                                                                        [mem_contexts, mem_outputs, mem_counter,
                                                                         tf.stack(output.decoder_target_list, axis=1), mem_output_input,
                                                                         output.contexts,
                                                                         tf.stack(rule_target_input_placeholder, axis=1),
                                                                         tf.stack(rule_id_input_placeholder, axis=1),
                                                                         self.global_step,
                                                                         emb_simple,
                                                                         output.encoder_outputs],
                                                                        [tf.float32, tf.float32, tf.int32],
                                                                        stateful=False, name='update_memory')

                #Loss and corresponding prior/mask
                decode_word_weight_list = [tf.to_float(tf.not_equal(d, self.data.vocab_simple.encode(constant.SYMBOL_PAD)))
                     for d in output.gt_target_list]
                decode_word_weight = tf.stack(decode_word_weight_list, axis=1)

                gt_target = tf.stack(output.gt_target_list, axis=1)

                def self_critical_loss():
                    # For minimize the negative log of probabilities
                    rewards = tf.py_func(self.metric.self_crititcal_reward,
                                         [sentence_idxs,
                                          tf.stack(output.sample_target_list, axis=-1),
                                          tf.stack(output.decoder_target_list, axis=-1),
                                          tf.stack(sentence_simple_input_placeholder, axis=-1),
                                          tf.stack(sentence_complex_input_placeholder, axis=-1),
                                          tf.ones((1,1)),
                                          # tf.stack(rule_target_input_placeholder, axis=1)
                                          ],
                                         tf.float32, stateful=False, name='reward')
                    rewards.set_shape((self.model_config.batch_size, self.model_config.max_simple_sentence))
                    rewards = tf.unstack(rewards, axis=1)

                    weighted_probs_list = [rewards[i] * decode_word_weight_list[i] * -output.sample_logit_list[i]
                                      for i in range(len(decode_word_weight_list))]
                    total_size = tf.reduce_sum(decode_word_weight_list)
                    total_size += 1e-12
                    weighted_probs = tf.reduce_sum(weighted_probs_list) / total_size
                    loss = weighted_probs
                    return loss

                def teacherforce_critical_loss():
                    losses = []
                    for step in range(self.model_config.max_simple_sentence):
                        logit = output.decoder_logit_list[step]
                        greedy_target_unit = tf.stop_gradient(tf.argmax(logit, axis=1))
                        if self.model_config.train_mode == 'teachercriticalv2':
                            sampled_target_unit, reward = tf.py_func(self.metric.self_crititcal_reward_unitv2,
                                                [sentence_idxs, step,
                                                 greedy_target_unit,
                                                 tf.stack(sentence_simple_input_placeholder, axis=-1),
                                                 tf.stack(sentence_complex_input_placeholder, axis=-1),
                                                 ],
                                                [tf.int32, tf.float32], stateful=False, name='reward')
                            reward.set_shape((self.model_config.batch_size,))
                            sampled_target_unit.set_shape((self.model_config.batch_size,))
                        elif self.model_config.train_mode == 'teachercritical':
                            sampled_target_unit = tf.cast(tf.squeeze(tf.multinomial(logit, 1), axis=1), tf.int32)
                            reward = tf.py_func(self.metric.self_crititcal_reward_unit,
                                                 [sentence_idxs, step,
                                                  sampled_target_unit, greedy_target_unit,
                                                  tf.stack(sentence_simple_input_placeholder, axis=-1),
                                                  tf.stack(sentence_complex_input_placeholder, axis=-1),
                                                  tf.ones((1, 1)),
                                                  ],
                                                 tf.float32, stateful=False, name='reward')
                            reward.set_shape((self.model_config.batch_size, ))
                        indices = tf.stack(
                            [tf.range(0, self.model_config.batch_size, dtype=tf.int32),
                             tf.squeeze(sampled_target_unit)],
                            axis=-1)
                        logit_unit = tf.gather_nd(tf.nn.softmax(logit, axis=1), indices)
                        decode_word_weight = decode_word_weight_list[step]
                        losses.append(-logit_unit * reward * decode_word_weight)
                    loss = tf.add_n(losses)
                    return loss

                def teacherforce_loss():
                    if self.model_config.number_samples > 0:
                        loss_fn = tf.nn.sampled_softmax_loss
                    else:
                        loss_fn = None
                    loss = sequence_loss(logits=tf.stack(output.decoder_logit_list, axis=1),
                                         targets=gt_target,
                                         weights=decode_word_weight,
                                         # softmax_loss_function=loss_fn,
                                         # w=w,
                                         # b=b,
                                         # decoder_outputs=decoder_outputs,
                                         # number_samples=self.model_config.number_samples
                                         )
                    return loss

                if self.model_config.train_mode == 'dynamic_self-critical':
                    loss = self_critical_loss()
                    # loss = tf.cond(
                    #     tf.greater(self.global_step, 50000),
                    #     # tf.logical_and(tf.greater(self.global_step, 100000), tf.equal(tf.mod(self.global_step, 2), 0)),
                    #     lambda : self_critical_loss(),
                    #     lambda : teacherforce_loss())
                elif self.model_config.train_mode == 'teachercritical' or self.model_config.train_mode == 'teachercriticalv2':
                    loss = tf.cond(
                        tf.equal(tf.mod(self.global_step, 3), 0),
                        lambda : teacherforce_loss(),
                        lambda : teacherforce_critical_loss())

                    # loss = teacherforce_critical_loss()
                else:
                    loss = teacherforce_loss()

                # if 'ruleattn' in self.model_config.external_loss:
                #     batch_pos = tf.range(
                #         self.model_config.batch_size * self.model_config.max_cand_rules) // self.model_config.max_cand_rules
                #     batch_pos = tf.reshape(
                #         batch_pos, [self.model_config.batch_size, self.model_config.max_cand_rules])
                #     batch_pos = tf.expand_dims(batch_pos, axis=2)
                #     ids = tf.stack(rule_pair_input_placeholder, axis=1)
                #     bias = 1.0 - tf.to_float(
                #         tf.logical_and(tf.equal(ids[:, :, 0], 0), tf.equal(ids[:, :, 1], 0)))
                #     ids = tf.concat([batch_pos, ids], axis=2)
                #     distrs = tf.stack(output.attn_distr_list, axis=1)
                #     ruleattn_loss = -tf.gather_nd(distrs, ids)*bias
                #     loss += ruleattn_loss
                #     self.pairs = tf.stack(rule_pair_input_placeholder, axis=1)

                obj = {
                    'sentence_idxs': sentence_idxs,
                    'sentence_simple_input_placeholder': sentence_simple_input_placeholder,
                    'sentence_complex_input_placeholder': sentence_complex_input_placeholder,
                }
                self.logits = output.decoder_logit_list
                if 'rule' in self.model_config.memory:
                    obj['rule_id_input_placeholder'] = rule_id_input_placeholder
                    obj['rule_target_input_placeholder'] = rule_target_input_placeholder
                    obj['rule_pair_input_placeholder'] = rule_pair_input_placeholder
                    obj['mem_contexts'] = mem_contexts
                    obj['mem_outputs'] = mem_outputs
                    obj['mem_counter'] = mem_counter
                return loss, obj

    def get_optim(self):
        learning_rate = tf.constant(self.model_config.learning_rate)

        if self.model_config.optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate)
        # Adam need lower learning rate
        elif self.model_config.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate)
        elif self.model_config.optimizer == 'lazy_adam':
            if not hasattr(self, 'hparams'):
                # In case not using Transformer model
                from tensor2tensor.models import transformer
                self.hparams = transformer.transformer_base()
            opt = tf.contrib.opt.LazyAdamOptimizer(
                self.hparams.learning_rate / 100.0,
                beta1=self.hparams.optimizer_adam_beta1,
                beta2=self.hparams.optimizer_adam_beta2,
                epsilon=self.hparams.optimizer_adam_epsilon)
        elif self.model_config.optimizer == 'adadelta':
            opt = tf.train.AdadeltaOptimizer(learning_rate)
        else:
            raise Exception('Not Implemented Optimizer!')

        # if self.model_config.max_grad_staleness > 0:
        #     opt = tf.contrib.opt.DropStaleGradientOptimizer(opt, self.model_config.max_grad_staleness)

        return opt

    # Got from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

class ModelOutput:
    def __init__(self, decoder_outputs_list=None, decoder_logit_list=None, decoder_target_list=None,
                 decoder_score=None, gt_target_list=None, encoder_embed_inputs_list=None, encoder_outputs=None,
                 contexts=None, final_outputs_list=None, sample_target_list=None, sample_logit_list=None, attn_distr_list=None):
        self._decoder_outputs_list = decoder_outputs_list
        self._decoder_logit_list = decoder_logit_list
        self._decoder_target_list = decoder_target_list
        self._decoder_score = decoder_score
        self._gt_target_list = gt_target_list
        self._encoder_embed_inputs_list = encoder_embed_inputs_list
        self._encoder_outputs = encoder_outputs
        self._contexts = contexts
        self._final_outputs_list = final_outputs_list
        self._sample_target_list = sample_target_list
        self._sample_logit_list = sample_logit_list
        self._attn_distr_list = attn_distr_list

    @property
    def encoder_outputs(self):
        return self._encoder_outputs

    @property
    def encoder_embed_inputs_list(self):
        """The final embedding input before model."""
        return self._encoder_embed_inputs_list

    @property
    def decoder_outputs_list(self):
        return self._decoder_outputs_list

    @property
    def final_outputs_list(self):
        return self._final_outputs_list

    @property
    def decoder_logit_list(self):
        return self._decoder_logit_list

    @property
    def decoder_target_list(self):
        return self._decoder_target_list

    @property
    def contexts(self):
        return self._contexts

    @property
    def decoder_score(self):
        return self._decoder_score

    @property
    def gt_target_list(self):
        return self._gt_target_list

    @property
    def sample_target_list(self):
        return self._sample_target_list

    @property
    def sample_logit_list(self):
        return self._sample_logit_list

    @property
    def attn_distr_list(self):
        return self._attn_distr_list
