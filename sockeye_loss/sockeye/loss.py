# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Functions to generate loss symbols for sequence-to-sequence models.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import mxnet as mx
from mxnet.metric import EvalMetric
from mxnet.symbol import broadcast_mul

from . import config
from . import constants as C

logger = logging.getLogger(__name__)

import pickle
import numpy as np

## ADDED CODE: gets simple probabilities from pickle file (generated beforehand)
def get_simple_probs(pickle_file, complexity_weight):
    with open(pickle_file, 'rb') as f:
        complex_preds = pickle.load(f)

    new_preds = []
    for p in complex_preds:
        if p == -1.0:
            new_preds.append(1.0)
        else:
            new_preds.append(4.0 - p + 1.0)
    ## Normalizes all probabilities
    new_preds = [(p/sum(new_preds)) ** complexity_weight for p in new_preds]
    print("LENGTH OF PREDICTIONS: " + str(len(new_preds)))
       
    return mx.nd.array(new_preds)

class LossConfig(config.Config):
    """
    Loss configuration.

    :param name: Loss name.
    :param vocab_size: Target vocab size.
    :param normalization_type: How to normalize the loss.
    :param label_smoothing: Optional smoothing constant for label smoothing.
    """

    def __init__(self,
                 name: str,
                 vocab_size: int,
                 normalization_type: str,
                 label_smoothing: float = 0.0, 
                 complexity_file: str = "NONE", 
                 complexity_weight: float = 0.0) -> None:
        super().__init__()

        self.name = name
        self.vocab_size = vocab_size
        self.normalization_type = normalization_type
        self.label_smoothing = label_smoothing
        self.complexity_file = complexity_file
        self.complexity_weight = complexity_weight


def get_loss(loss_config: LossConfig) -> 'Loss':
    """
    Returns Loss instance.

    :param loss_config: Loss configuration.
    """
    if loss_config.name == C.CROSS_ENTROPY:
        return CrossEntropyLoss(loss_config)

    ## This is where we need to put it!
    elif loss_config.name == C.SIMPLE_CROSS_ENTROPY:
        return SimpleCrossEntropyLoss(loss_config)
    else:
        raise ValueError("unknown loss name: %s" % loss_config.name)


class Loss(ABC):
    """
    Generic Loss interface.
    get_loss() method should return a loss symbol and the softmax outputs.
    The softmax outputs (named C.SOFTMAX_NAME) are used by EvalMetrics to compute various metrics,
    e.g. perplexity, accuracy. In the special case of cross_entropy, the SoftmaxOutput symbol
    provides softmax outputs for forward() AND cross_entropy gradients for backward().
    """

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss and softmax output symbols.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_metric(self) -> EvalMetric:
        """
        Create an instance of the EvalMetric that corresponds to this Loss function.
        """
        pass


class CrossEntropyLoss(Loss):
    """
    Computes the cross-entropy loss.

    :param loss_config: Loss configuration.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: CrossEntropy(normalization_type=%s, label_smoothing=%s)",
                    loss_config.normalization_type, loss_config.label_smoothing)
        self.loss_config = loss_config

    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss symbol.
        """
        if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
            normalization = "valid"
        elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
            normalization = "null"
        else:
            raise ValueError("Unknown loss normalization type: %s" % self.loss_config.normalization_type)

        return [mx.sym.SoftmaxOutput(data=logits,
                                     label=labels,
                                     ignore_label=C.PAD_ID,
                                     use_ignore=True,
                                     normalization=normalization,
                                     smooth_alpha=self.loss_config.label_smoothing,
                                     name=C.SOFTMAX_NAME)]

    def create_metric(self) -> "CrossEntropyMetric":
        return CrossEntropyMetric(self.loss_config)

class SimpleCrossEntropyLoss(Loss):
    """
    Computes a simplified cross-entropy loss. 
    :param alpha: Smoothing value.
    :param vocab_size: Size of the target vocabulary.
    :param normalize: If True normalize the gradient by dividing by the number of non-PAD tokens.
    """

    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: SimpleCrossEntropy(normalization_type=%s, label_smoothing=%s)",
                    loss_config.normalization_type, loss_config.label_smoothing)
        self.loss_config = loss_config
        self._alpha = loss_config.label_smoothing
        self._vocab_size = loss_config.vocab_size
        self._normalize = loss_config.normalization_type
        self._complexity_file = loss_config.complexity_file
        self._complexity_weight = loss_config.complexity_weight

        complex_preds = mx.nd.array([])
        if self._complexity_file != "NONE":
            complex_preds = get_simple_probs(self._complexity_file, self._complexity_weight)
        self._complex_preds = complex_preds


    def get_loss(self, logits: mx.sym.Symbol, labels: mx.sym.Symbol) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.
        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss and softmax output symbols.
        """
        probs = mx.sym.softmax(data=logits)

        on_value = 1.0 - self._alpha
        off_value = self._alpha / (self._vocab_size - 1.0)
        cross_entropy = mx.sym.one_hot(indices=mx.sym.cast(data=labels, dtype='int32'),
                                       depth=self._vocab_size,
                                       on_value=on_value,
                                       off_value=off_value)
        # zero out pad symbols (0)
        cross_entropy = mx.sym.where(labels, cross_entropy, mx.sym.zeros((0, self._vocab_size)))

        ## ADDED CODE: makes a symbol of the complexity weights
        #weights = mx.sym.Variable('weights', shape = (len(self._complex_preds)), init = mx.init.One())
        w = mx.sym.Variable('w')
        w = mx.sym.BlockGrad(w)
        weights = (w).bind(mx.cpu(), {'w': self._complex_preds})

        #weights.set_params(self._complex_preds)

        ## ADDED CODE: does element-wise multiplication
        weighted_probs = broadcast_mul(probs, weights)
        cross_entropy = - mx.sym.log(data=weighted_probs) * cross_entropy
        
        #cross_entropy = - mx.sym.log(data=probs + 1e-10) * cross_entropy
        cross_entropy = mx.sym.sum(data=cross_entropy, axis=1)

        cross_entropy = mx.sym.MakeLoss(cross_entropy, name=C.SIMPLE_CROSS_ENTROPY)
        probs = mx.sym.BlockGrad(probs, name=C.SOFTMAX_NAME)
        return [cross_entropy, probs]

    def create_metric(self) -> "CrossEntropyMetric":
        return CrossEntropyMetric(self.loss_config)


class CrossEntropyMetric(EvalMetric):
    """
    Version of the cross entropy metric that ignores padding tokens.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """

    def __init__(self,
                 loss_config: LossConfig,
                 name: str = C.CROSS_ENTROPY,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=output_names, label_names=label_names)
        self.loss_config = loss_config

    @staticmethod
    def cross_entropy(logprob, label):
        ce = -mx.nd.pick(logprob, label)  # pylint: disable=invalid-unary-operand-type
        return ce

    @staticmethod
    def cross_entropy_smoothed(logprob, label, alpha, num_classes):
        ce = CrossEntropyMetric.cross_entropy(logprob, label)
        # gain for each incorrect class
        per_class_gain = alpha / (num_classes - 1)
        # discounted loss for correct class
        ce *= 1 - alpha - per_class_gain
        # add gain for incorrect classes to total cross-entropy
        ce -= mx.nd.sum(logprob * per_class_gain, axis=-1, keepdims=False)
        return ce

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            batch_size = label.shape[0]
            label = label.as_in_context(pred.context).reshape((label.size,))

            logprob = mx.nd.log(mx.nd.maximum(1e-10, pred))

            # ce: (batch*time,)
            if self.loss_config.label_smoothing > 0.0:
                ce = self.cross_entropy_smoothed(logprob, label,
                                                 alpha=self.loss_config.label_smoothing,
                                                 num_classes=self.loss_config.vocab_size)
            else:
                ce = self.cross_entropy(logprob, label)

            # mask pad tokens
            valid = (label != C.PAD_ID).astype(dtype=pred.dtype)
            ce *= valid

            ce = mx.nd.sum(ce)
            if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
                num_valid = mx.nd.sum(valid)
                ce /= num_valid
                self.num_inst += 1
            elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
                # When not normalizing, we divide by the batch size (number of sequences)
                # NOTE: This is different from MXNet's metrics
                self.num_inst += batch_size

            self.sum_metric += ce.asscalar()
