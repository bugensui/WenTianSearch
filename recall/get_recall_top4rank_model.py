"""
1. 为了排序模型准备数据对
2. 不要把官方数据和外部数据混在一块计算，一定要分开计算
"""

import os
import csv
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

import faiss
# sudo apt-get install libopenblas-dev
# sudo apt-get install libomp-dev
from transformers import AutoTokenizer

# 外部包
from utils import set_seeds
from get_recall_top4hard_neg import get_embed, get_id2id, get_id2text, build_index

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")


def gen_save_data_pair(D, I, q_ids, save_path, write_model, external_data):
    data_pairs = []
    for i in range(len(I)):
        probs = D[i]
        query_id = q_ids[i]
        recall_id = I[i]
        recall_id = [d+1 for d in recall_id]
        assert len(probs) == len(recall_id)
        
        # 防止伪负例，分三步（掐top2；掐相似度高于0.2的；剩下部分采样20个）
        
        ## 掐top2
        recall_id = recall_id[10:]
        probs = probs[10:]        
        
        label_id = id_q2d[query_id]

        new_recall_id = []
        for d in recall_id:
            if d == label_id: continue
            if external_data: new_recall_id.append(str(d+1001500))
            else: new_recall_id.append(str(d))
        recall_id = new_recall_id
        if external_data:
            query_id += 100000
            label_id += 1001500
            data_pair = str(query_id) + '\t' + str(label_id) + '\t' + "#".join(recall_id)
        else:
            data_pair = str(query_id) + '\t' + str(label_id) + '\t' + "#".join(recall_id)

        data_pairs.append(data_pair)
    print("data_pairs", data_pairs[:10])

    if external_data:
        with open(save_path, write_model) as f:
            f.write("\n".join(data_pairs))
    else:
        with open(save_path, write_model) as f:
            f.write("\n".join(data_pairs)+"\n")


if __name__ == "__main__":
    set_seeds(2022)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_embed_path", type=str, default="./query_embedding", help="query_embedding的路径") 
    parser.add_argument("--doc_embed_path", type=str, default="./doc_embedding", help="doc_embedding的路径") 
    parser.add_argument("--qrels_path", type=str, default="./data/qrels.train.tsv", help="qrels的路径") 
    parser.add_argument("--corpus_path", type=str, default="./data/corpus.tsv", help="corpus的路径")
    parser.add_argument("--topk", type=int, default=10, help="召回topk")
    parser.add_argument("--save_path", type=str, default=None, help="rank模型训练数据保存路径， None表示不做该操作")
    parser.add_argument("--index_flat_mode", type=str, default="l2", help="相似度计算方式")
    parser.add_argument("--write_model", type=str, default="w", help="数据对的写入方式")
    parser.add_argument("--external_data", action='store_true', default=False, help="是否是外部数据")
    
    args = parser.parse_args()
    print(args) 
    
    # 获取embedding等信息
    q_ids, q_embeds, q_id2tokens = get_embed(args.query_embed_path)
    d_ids, d_embeds, d_id2tokens = get_embed(args.doc_embed_path)

    # 获取各种映射关系
    id_q2d, id_d2q, doc_list = get_id2id(args.qrels_path)
    did2text = get_id2text(args.corpus_path)

    # 开始检索
    index = build_index(d_embeds, d_ids, args.index_flat_mode)
    D, I = index.search(q_embeds, args.topk)

    # 生成训练用的数据对
    gen_save_data_pair(D, I, q_ids, args.save_path, args.write_model, args.external_data)

