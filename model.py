# -*- coding: utf-8 -*-
"""
@author: kebo
@contact: kebo0912@outlook.com
@version: 1.0

@file: model.py 
@time: 2020/6/3 下午11:45

这一行开始写关于本文件的说明与解释

"""
from logging import getLogger
import tensorflow as tf

from utils import dot

logger = getLogger(__name__)


class GraphConvolution(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0,
                 is_sparse_inputs=False,
                 activation=tf.nn.relu,
                 use_bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution).__init__(**kwargs)

        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.use_bias = use_bias
        self.num_features_nonzero = num_features_nonzero

        self.weights_ = []
        for i in range(1):
            w = self.add_variable('weight' + str(i), [input_dim, output_dim])
            self.weights_.append(w)
        if self.use_bias:
            self.bias = self.add_variable('bias', [output_dim])

        # for p in self.trainable_variables:
        #     print(p.name, p.shape)

    def call(self, inputs, training=False, **kwargs):
        x, support_ = inputs

        # dropout
        if training and self.is_sparse_inputs:
            x = self.sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # convolve
        supports = list()
        for i in range(len(support_)):
            if not self.featureless:  # if it has features x
                pre_sup = dot(x, self.weights_[i], spares=self.is_sparse_inputs)
            else:
                pre_sup = self.weights_[i]
            support = dot(support_[i], pre_sup, spares=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.bias

        return self.activation(output)

    @classmethod
    def sparse_dropout(cls, x, rate, noise_shape):
        """
        Dropout for sparse tensors.
        """
        random_tensor = 1 - rate
        random_tensor += tf.random.uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse.retain(x, dropout_mask)
        return pre_out * (1. / (1 - rate))


class GCN(tf.keras.models.Model):
    def __init__(self, input_dim, output_dim, num_features_nonzero, **kwargs):
        super(GCN).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features_nonzero = num_features_nonzero

        logger.info('input dim: ', input_dim)
        logger.info('output dim:', output_dim)
        logger.info('num_features_nonzero: ', num_features_nonzero)

        self.layers_ = []
        self.layers_.append()

    def call(self, inputs, training=None, mask=None):
        x, label, mask, support = inputs
