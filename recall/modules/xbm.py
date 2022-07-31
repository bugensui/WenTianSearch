# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch


class XBM(object):
    def __init__(self, size=8196, weight=1, start_step=2000):
        self.K = size   # 缓存多少样本
        self.weight = weight  # xbm的loss所占权重
        self.start_step = start_step  # 迭代多少次后才开始执行xbm
        self.query_feats = torch.zeros(self.K, 128).cuda()  # 缓存query样本特征
        self.doc_feats = torch.zeros(self.K, 128).cuda()  # 缓存query对应doc样本特征
        self.neg_feats = torch.zeros(self.K, 128).cuda()  # 缓存query对应难样本特征
        self.ptr = 0  # cross_batch 当前缓存的量
        self.is_full = False  # 缓存是否满了
        self.is_neg = False  # 是否包括难负样例 

    def get(self):
        if self.is_full:
            res = (self.query_feats, self.doc_feats, self.neg_feats) if self.is_neg else (self.query_feats, self.doc_feats)
            return res
        else:
            res = (self.query_feats[:self.ptr], self.doc_feats[:self.ptr], self.neg_feats[:self.ptr]) if self.is_neg else (self.query_feats[:self.ptr], self.doc_feats[:self.ptr])
            return res
           
    def enqueue_dequeue(self, query_feats, doc_feats, neg_feats=None):
        q_size = len(doc_feats)
        if self.ptr + q_size > self.K:   # 剩余缓存不够存的
            self.query_feats[-q_size:] = query_feats
            self.doc_feats[-q_size:] = doc_feats
            self.ptr = 0
            self.is_full = True
            if neg_feats:
                self.neg_feats[-q_size:] = neg_feats
                self.is_neg = True

        else:
            self.query_feats[self.ptr: self.ptr + q_size] = query_feats
            self.doc_feats[self.ptr: self.ptr + q_size] = doc_feats
            self.ptr += q_size
            if neg_feats:
                self.neg_feats[self.ptr: self.ptr + q_size] = neg_feats
                self.is_neg = True
            