# -*- coding: utf-8 -*-
"""
@author: kebo
@contact: kebo0912@outlook.com
@version: 1.0

@file: utils.py 
@time: 2020/7/7 下午11:36

这一行开始写关于本文件的说明与解释

"""
import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """uniform init."""
    initial = tf.random.uniform(shape, minval=-scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dot(x, y, spares=False):
    """
    wrapper for tf.matmul (sparse vs dense)
    :param x:
    :param y:
    :param spares:
    :return:
    """
    if spares:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res
