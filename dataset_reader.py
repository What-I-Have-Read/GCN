# -*- coding: utf-8 -*-
"""
@author: kebo
@contact: itachi971009@gmail.com
@version: 1.0

@file: dataset_reader.py 
@time: 2020/6/3 下午11:57

这一行开始写关于本文件的说明与解释

"""
import os
import pickle
import numpy as np
import networkx as nx
import scipy.sparse as sp
from typing import List


class DatasetReader:
    def __init__(self, data_path: str, dataset_str: str):
        self.data_path = data_path
        self.dataset_str = dataset_str

    def read(self):
        """
        Loads input data from data directory

        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :return:
        """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(os.path.join(self.data_path, "ind.{}.{}".format(self.dataset_str, names[i]), 'rb')) as f:
                objects.append(pickle.load(f, encoding='latin1'))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = self.parse_index_file(
            os.path.join(self.data_path, "ind.{}.test.index".format(self.dataset_str)))
        test_idx_range = np.sort(test_idx_reorder)
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = self.sample_mask(idx_train, labels.shape[0])
        val_mask = self.sample_mask(idx_val, labels.shape[0])
        test_mask = self.sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

    @classmethod
    def parse_index_file(cls, filename: str) -> List:
        """
         Parse index file.
        :param filename:
        :return:
        """
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    @classmethod
    def sample_mask(cls, idx, length):
        """
        Create mask.
        """
        mask = np.zeros(length)
        mask[idx] = 1
        return np.array(mask, dtype=np.bool)
