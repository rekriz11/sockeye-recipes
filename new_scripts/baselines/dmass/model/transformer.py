import tensorflow as tf
from tensor2tensor.layers import common_attention, common_layers
from tensor2tensor.models import transformer
from tensor2tensor.models.transformer import transformer_ffn_layer
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import beam_search_backprop, beam_search_backprop_sampled, beam_search_backprop_greedy

from util import constant
from model.graph import Graph
from model.graph import ModelOutput
from util.nn import linear_3d, linear


class TransformerGraph(Graph):
    def __init__(self, data, is_train, model_config):
        super(TransformerGraph, self).__init__(data, is_train, model_config)
        self.hparams = transformer.transformer_base()
        self.setup_hparams()
        self.model_fn = self.transformer_fn

    def transformer_fn(self,
                       sentence_complex_input_placeholder, emb_complex,
                       sentence_simple_input_placeholder, emb_simple,
                       w, b,
                       rule_id_input_placeholder, mem_contexts, mem_outputs,
                       global_step):
        encoder_embed_inputs = tf.stack(
            self.embedding_fn(sentence_complex_input_placeholder, emb_complex), axis=1)
        encoder_attn_bias = common_attention.attention_bias_ignore_padding(
            tf.to_float(tf.equal(tf.stack(sentence_complex_input_placeholder, axis=1),
                                 self.data.vocab_complex.encode(constant.SYMBOL_PAD))))
        if self.hparams.pos == 'timing':
            encoder_embed_inputs = common_attention.add_timing_signal_1d(encoder_embed_inputs)
            print('Use positional encoding in encoder text.')

        with tf.variable_scope('transformer_encoder'):
            encoder_embed_inputs = tf.nn.dropout(encoder_embed_inputs,
                                                 1.0 - self.hparams.layer_prepostprocess_dropout)
            encoder_outputs = transformer.transformer_encoder(
                encoder_embed_inputs, encoder_attn_bias, self.hparams)

        encoder_embed_inputs_list = tf.unstack(encoder_embed_inputs, axis=1)
        with tf.variable_scope('transformer_decoder'):
            train_mode = self.model_config.train_mode
            if self.is_train and (train_mode == 'teacher' or
                                  train_mode == 'teachercritical'or train_mode ==  'teachercriticalv2'):
                # General train
                print('Use Generally Process.')
                decoder_embed_inputs_list = self.embedding_fn(
                    sentence_simple_input_placeholder[:-1], emb_simple)
                final_output_list, decoder_output_list, cur_context = self.decode_step(
                    decoder_embed_inputs_list, encoder_outputs, encoder_attn_bias,
                    rule_id_input_placeholder, mem_contexts, mem_outputs, global_step)
                decoder_logit_list = [self.output_to_logit(o, w, b) for o in final_output_list]
                decoder_target_list = [tf.argmax(o, output_type=tf.int32, axis=-1)
                                       for o in decoder_logit_list]
            elif self.is_train and train_mode == 'dynamic_self-critical':
                decoder_target_tensor = tf.TensorArray(tf.int32, size=0, dynamic_size=True,
                                                       clear_after_read=False,
                                                       element_shape=[self.model_config.batch_size, ])
                sampled_target_tensor = tf.TensorArray(tf.int32, size=0, dynamic_size=True,
                                                      clear_after_read=False,
                                                      element_shape=[self.model_config.batch_size, ])
                sampled_logit_tensor = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                                                     clear_after_read=False,
                                                     element_shape=[self.model_config.batch_size, ])

                def _is_finished(step, decoder_target_tensor, sampled_target_tensor, sampled_logit_tensor):
                    return tf.less(step, self.model_config.max_simple_sentence)

                def _recursive(step, decoder_target_tensor, sampled_target_tensor, sampled_logit_tensor):
                    decoder_target_stack = tf.transpose(decoder_target_tensor.stack(), perm=[1, 0])

                    def get_empty_emb():
                        decoder_emb_inputs = tf.zeros(
                            [self.model_config.batch_size, 1, self.model_config.dimension])
                        return decoder_emb_inputs
                    def get_emb():
                        batch_go = tf.zeros(
                            [self.model_config.batch_size, 1, self.model_config.dimension])
                        decoder_emb_inputs = tf.concat([
                            batch_go, tf.gather(emb_simple, decoder_target_stack)], axis=1)
                        return decoder_emb_inputs

                    decoder_emb_inputs = tf.cond(tf.equal(step, 0), lambda :get_empty_emb(), lambda :get_emb())

                    final_outputs, _, _ = self.decode_inputs_to_outputs(
                        decoder_emb_inputs, encoder_outputs, encoder_attn_bias,
                        rule_id_input_placeholder, mem_contexts, mem_outputs, global_step)
                    final_output = final_outputs[:, -1, :]
                    decoder_logit = tf.add(tf.matmul(final_output, tf.transpose(w)), b)
                    decoder_target = tf.stop_gradient(tf.argmax(decoder_logit, output_type=tf.int32, axis=-1))
                    sampled_target = tf.cast(tf.squeeze(tf.multinomial(decoder_logit, 1), axis=1), tf.int32)

                    indices = tf.stack(
                        [tf.range(0, self.model_config.batch_size, dtype=tf.int32),
                         tf.squeeze(sampled_target)],
                        axis=-1)
                    logit_unit = tf.gather_nd(tf.nn.softmax(decoder_logit, axis=1), indices)

                    decoder_target_tensor = decoder_target_tensor.write(step, decoder_target)
                    sampled_target_tensor = sampled_target_tensor.write(step, sampled_target)
                    sampled_logit_tensor = sampled_logit_tensor.write(step, logit_unit)

                    return step+1, decoder_target_tensor, sampled_target_tensor, sampled_logit_tensor


                step = tf.constant(0)
                (_, decoder_target_tensor, sampled_target_tensor, sampled_logit_tensor) = tf.while_loop(
                    _is_finished, _recursive,
                    [step, decoder_target_tensor, sampled_target_tensor, sampled_logit_tensor],
                    back_prop=True, parallel_iterations=1, swap_memory=False)

                decoder_target_tensor = decoder_target_tensor.stack()
                decoder_target_tensor.set_shape([self.model_config.max_simple_sentence,
                                                 self.model_config.batch_size])
                decoder_target_tensor = tf.transpose(decoder_target_tensor, perm=[1, 0])
                decoder_target_list = tf.unstack(decoder_target_tensor, axis=1)


                sampled_target_tensor = sampled_target_tensor.stack()
                sampled_target_tensor.set_shape([self.model_config.max_simple_sentence,
                                                self.model_config.batch_size])
                sampled_target_tensor = tf.transpose(sampled_target_tensor, perm=[1, 0])
                sampled_target_list = tf.unstack(sampled_target_tensor, axis=1)

                sampled_logit_tensor = sampled_logit_tensor.stack()
                sampled_logit_tensor.set_shape([self.model_config.max_simple_sentence,
                                               self.model_config.batch_size])
                sampled_logit_tensor = tf.transpose(sampled_logit_tensor, perm=[1, 0])
                sampled_logit_list = tf.unstack(sampled_logit_tensor, axis=1)

            else:
                # Beam Search
                print('Use Beam Search with Beam Search Size %d.' % self.model_config.beam_search_size)
                return self.transformer_beam_search(encoder_outputs, encoder_attn_bias, encoder_embed_inputs_list,
                                                    sentence_complex_input_placeholder, emb_simple, w, b,
                                                    rule_id_input_placeholder, mem_contexts, mem_outputs, global_step)

        gt_target_list = sentence_simple_input_placeholder
        output = ModelOutput(
            contexts=cur_context if 'rule' in self.model_config.memory else None,
            encoder_outputs=encoder_outputs,
            decoder_outputs_list=final_output_list if train_mode != 'dynamic_self-critical' else None,
            final_outputs_list=final_output_list if train_mode != 'dynamic_self-critical' else None,
            decoder_logit_list=decoder_logit_list if train_mode != 'dynamic_self-critical' else None,
            gt_target_list=gt_target_list,
            encoder_embed_inputs_list=tf.unstack(encoder_embed_inputs, axis=1),
            decoder_target_list=decoder_target_list,
            sample_logit_list=sampled_logit_list if train_mode == 'dynamic_self-critical' else None,
            sample_target_list=sampled_target_list if train_mode == 'dynamic_self-critical' else None
        )
        return output

    def decode_step(self, decode_input_list, encoder_outputs, encoder_attn_bias,
                    rule_id_input_placeholder, mem_contexts, mem_outputs, global_step):
        batch_go = [tf.zeros([self.model_config.batch_size, self.model_config.dimension])]
        target_length = len(decode_input_list) + 1
        decoder_emb_inputs = tf.stack(batch_go + decode_input_list, axis=1)
        final_output, decoder_output, cur_context = self.decode_inputs_to_outputs(
            decoder_emb_inputs, encoder_outputs, encoder_attn_bias,
            rule_id_input_placeholder, mem_contexts, mem_outputs, global_step)

        decoder_output_list = [
            tf.squeeze(d, 1)
            for d in tf.split(decoder_output, target_length, axis=1)]
        final_output_list = [
            tf.squeeze(d, 1)
            for d in tf.split(final_output, target_length, axis=1)]
        return final_output_list, decoder_output_list, cur_context

    def transformer_beam_search(self, encoder_outputs, encoder_attn_bias, encoder_embed_inputs_list,
                                sentence_complex_input_placeholder, emb_simple, w, b,
                                rule_id_input_placeholder, mem_contexts, mem_outputs, global_step):
        # Use Beam Search in evaluation stage
        # Update [a, b, c] to [a, a, a, b, b, b, c, c, c] if beam_search_size == 3
        encoder_beam_outputs = tf.concat(
            [tf.tile(tf.expand_dims(encoder_outputs[o, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        encoder_attn_beam_bias = tf.concat(
            [tf.tile(tf.expand_dims(encoder_attn_bias[o, :, :, :], axis=0),
                     [self.model_config.beam_search_size, 1, 1, 1])
             for o in range(self.model_config.batch_size)], axis=0)

        def symbol_to_logits_fn(ids):
            cur_ids = ids[:, 1:]
            embs = tf.nn.embedding_lookup(emb_simple, cur_ids)
            embs = tf.pad(embs, [[0, 0], [1, 0], [0, 0]])
            final_outputs, _, _ = self.decode_inputs_to_outputs(embs, encoder_beam_outputs, encoder_attn_beam_bias,
                                                                rule_id_input_placeholder, mem_contexts, mem_outputs,
                                                                global_step)

            decoder_logit_list = self.output_to_logit(final_outputs[:, -1, :], w, b)
            return decoder_logit_list

        beam_ids, beam_score = beam_search.beam_search(symbol_to_logits_fn,
                                                       tf.zeros([self.model_config.batch_size], tf.int32),
                                                       self.model_config.beam_search_size,
                                                       self.model_config.max_simple_sentence,
                                                       self.data.vocab_simple.vocab_size(),
                                                       self.model_config.penalty_alpha,
                                                       )
        top_beam_ids = beam_ids[:, 0, 1:]
        top_beam_ids = tf.pad(top_beam_ids,
                              [[0, 0],
                               [0, self.model_config.max_simple_sentence - tf.shape(top_beam_ids)[1]]])
        decoder_target_list = [tf.squeeze(d, 1)
                               for d in tf.split(top_beam_ids, self.model_config.max_simple_sentence, axis=1)]
        decoder_score = -beam_score[:, 0] / tf.to_float(tf.shape(top_beam_ids)[1])

        # Get outputs based on target ids
        decode_input_embs = tf.stack(self.embedding_fn(decoder_target_list, emb_simple), axis=1)
        tf.get_variable_scope().reuse_variables()
        final_outputs, decoder_outputs, _ = self.decode_inputs_to_outputs(decode_input_embs, encoder_outputs, encoder_attn_bias,
                                                                          rule_id_input_placeholder, mem_contexts,
                                                                          mem_outputs, global_step)
        output = ModelOutput(
            encoder_outputs=encoder_outputs,
            final_outputs_list=final_outputs,
            decoder_outputs_list=decoder_outputs,
            decoder_score=decoder_score,
            decoder_target_list=decoder_target_list,
            encoder_embed_inputs_list=encoder_embed_inputs_list
        )
        return output

    def output_to_logit(self, prev_out, w, b):
        prev_logit = tf.add(tf.matmul(prev_out, tf.transpose(w)), b)
        return prev_logit

    def decode_inputs_to_outputs(self, decoder_embed_inputs, encoder_outputs, encoder_attn_bias,
                                 rule_id_input_placeholder, mem_contexts, mem_outputs, global_step):
        if self.hparams.pos == 'timing':
            decoder_embed_inputs = common_attention.add_timing_signal_1d(decoder_embed_inputs)
            print('Use positional encoding in decoder text.')

        decoder_attn_bias = common_attention.attention_bias_lower_triangle(tf.shape(decoder_embed_inputs)[1])
        decoder_embed_inputs = tf.nn.dropout(decoder_embed_inputs,
                                             1.0 - self.hparams.layer_prepostprocess_dropout)

        if 'rule' in self.model_config.memory:
            decoder_output, contexts = transformer.transformer_decoder2(
                decoder_embed_inputs, encoder_outputs, decoder_attn_bias,
                encoder_attn_bias, self.hparams)

            # encoder_gate_w = tf.get_variable('encoder_gate_w', shape=(
            #     1, self.model_config.dimension, 1))
            # encoder_gate_b = tf.get_variable('encoder_gate_b', shape=(1, 1, 1))
            # encoder_gate = tf.tanh(encoder_gate_b + tf.nn.conv1d(encoder_outputs, encoder_gate_w, 1, 'SAME'))
            # encoder_context_outputs = tf.expand_dims(tf.reduce_mean(encoder_outputs * encoder_gate, axis=1), axis=1)
            cur_context = contexts[0] #tf.concat(contexts, axis=-1)
            cur_mem_contexts = tf.stack(self.embedding_fn(rule_id_input_placeholder, mem_contexts), axis=1)
            cur_mem_outputs = tf.stack(self.embedding_fn(rule_id_input_placeholder, mem_outputs), axis=1)

            bias = tf.expand_dims(
                -1e9 * tf.to_float(tf.equal(tf.stack(rule_id_input_placeholder, axis=1), 0)),
                axis=1)
            weights = tf.nn.softmax(bias + tf.matmul(cur_context, cur_mem_contexts, transpose_b=True))
            mem_output = tf.matmul(weights, cur_mem_outputs)

            temp_output = tf.concat((decoder_output, mem_output), axis=-1)
            w = tf.get_variable('w_ffn', shape=(
                1, self.model_config.dimension*2, self.model_config.dimension))
            # b = tf.get_variable('b_ffn', shape=(
            #     1, 1, self.model_config.dimension))
            mem_output = tf.nn.conv1d(temp_output, w, 1, 'SAME')
            g = tf.greater(global_step, tf.constant(2*self.model_config.memory_prepare_step, dtype=tf.int64))
            final_output = tf.cond(g, lambda: mem_output, lambda: decoder_output)
            return final_output, decoder_output, cur_context
        else:
            decoder_output = transformer.transformer_decoder(
                decoder_embed_inputs, encoder_outputs, decoder_attn_bias,
                encoder_attn_bias, self.hparams)
            final_output = decoder_output
            return final_output, decoder_output, None

    def setup_hparams(self):
        self.hparams.num_heads = self.model_config.num_heads
        self.hparams.num_hidden_layers = self.model_config.num_hidden_layers
        self.hparams.num_encoder_layers = self.model_config.num_encoder_layers
        self.hparams.num_decoder_layers = self.model_config.num_decoder_layers
        self.hparams.pos = self.model_config.hparams_pos
        self.hparams.hidden_size = self.model_config.dimension
        self.hparams.layer_prepostprocess_dropout = self.model_config.layer_prepostprocess_dropout

        if self.is_train:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.TRAIN)
        else:
            self.hparams.add_hparam('mode', tf.estimator.ModeKeys.EVAL)
            self.hparams.layer_prepostprocess_dropout = 0.0
            self.hparams.attention_dropout = 0.0
            self.hparams.dropout = 0.0
            self.hparams.relu_dropout = 0.0

