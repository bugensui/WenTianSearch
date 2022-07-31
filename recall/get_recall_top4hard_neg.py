import os
import csv
import sys
import json
from random import shuffle
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import time

import faiss
# !apt-get install libopenblas-dev
# !apt-get install libomp-dev
from transformers import AutoTokenizer

# 外部包
from utils import set_seeds

tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")


def get_embed(embed_path):
    """获取embed信息
    """
    reader = csv.reader(open(embed_path), delimiter='\t')
    reader = [line for line in reader]
    q_ids, q_embeds, q_token_ids = [], [], []
    id2tokens = {}
    for line in tqdm(reader):
        q_id = int(line[0])

        q_embed = np.array(line[1].split(","))
        q_embed = np.array([float(em) for em in q_embed])
        
        q_token_id = np.array(line[2].split(","))

        q_ids.append(q_id)
        q_embeds.append(q_embed)
        q_token_ids.append(q_token_id)
        id2tokens[q_id] = q_token_id

    q_ids = np.array(q_ids)
    q_embeds = np.array(q_embeds).astype(np.float32)
    
    print("数目", q_embeds.shape)
    return q_ids, q_embeds, id2tokens


def get_id2id(path):
    """获取train_query到doc的映射（编号）
    """
    id_d2q = {}
    id_q2d = {}
    doc_list = []
    reader = csv.reader(open(path), delimiter='\t')
    for ind, line in enumerate(reader):
        if ind < 3:
            print(line)
        id_q = int(line[0])
        id_d = int(line[2])
        id_d2q[id_d] = id_q
        id_q2d[id_q] = id_d
        doc_list.append(id_d)
    
    return id_q2d, id_d2q, doc_list


def get_id2text(path):
    """获取id到doc的映射关系
    """
    id2text = {}
    reader = csv.reader(open(path), delimiter='\t')
    for ind, line in enumerate(reader):
        if ind < 3:
            print(line)
        id = int(line[0])
        text = line[1]
        id2text[id] = text
    
    return id2text


def build_index(d_embeds, d_ids, index_flat_mode):
    """为doc建立索引
    """
    dim = 128
    
    if index_flat_mode == "l2": index = faiss.IndexFlatL2(dim)
    elif index_flat_mode == "ip": index = faiss.IndexFlatIP(dim)
    index2 = faiss.IndexIDMap(index)
    index2.add_with_ids(d_embeds, d_ids)
    
    return index

def save_hard_neg(D, I, doc_list, id_d2q, save_path):
    doc2simdoc = {}
    # 开始保存
    for i in range(len(D)):
        # 获取原始query
        simdocs = I[i]
        probs = D[i]
        temp = doc_list[i]
        doc = id_d2q[temp]

        # 去掉原始query
        simdocs = I[i][1:]
        probs = D[i][1:]

        # 找候选项
        candidates = []
        for simdoc, prob in zip(simdocs, probs):
            if prob <= 0.7:
                candidates.append(simdoc+1)
        if len(candidates) == 0:
            candidates.append(simdocs[-1]+1)

        # 随机选一个
        shuffle(candidates)
        doc2simdoc[str(doc)] = str(candidates[0])

    with open(save_path, "w") as f:
        json.dump(doc2simdoc, f)

def comput_local_recall(I, q_ids, id_q2d, k):
    recall_k = 0
    data_pairs = []
    for i in range(len(I)):
        query_id = q_ids[i]
        recall_id = I[i]
        recall_id = [d+1 for d in recall_id][:k]
        label_id = id_q2d[query_id]

        for d in recall_id:
            if d == label_id:
                recall_k += 1
    recall_k /= len(I)
    print(k, recall_k)
    with open("comput_recall_k_result.txt", "a+") as f:
        f.write(str(k) + "\t" + str(recall_k) + "\n")

        

if __name__ == "__main__":
    set_seeds(2022)

    parser = argparse.ArgumentParser()
    parser.add_argument("--query_embed_path", type=str, default="./query_embedding", help="query_embedding的路径") 
    parser.add_argument("--doc_embed_path", type=str, default="./doc_embedding", help="doc_embedding的路径") 
    parser.add_argument("--qrels_path", type=str, default="./data/qrels.train.tsv", help="qrels的路径") 
    parser.add_argument("--corpus_path", type=str, default="./data/corpus.tsv", help="corpus的路径")
    parser.add_argument("--topk", type=int, default=10, help="召回topk")
    parser.add_argument("--query_num", type=int, default=10, help="query请求检索量, -1表示全量")
    parser.add_argument("--save_path", type=str, default=None, help="难负例保存路径， None表示不做该操作")
    parser.add_argument("--local_recall_topk", action='store_true', default=False, help="计算本地召回topk的召回率，false表示不做该操作")
    parser.add_argument("--index_flat_mode", type=str, default="l2", help="相似度计算方式")
    
    
    
    args = parser.parse_args()
    print(args) 
    
    # 获取embedding等信息
    q_ids, q_embeds, q_id2tokens = get_embed(args.query_embed_path)
    d_ids, d_embeds, d_id2tokens = get_embed(args.doc_embed_path)

    # 获取各种映射关系
    id_q2d, id_d2q, doc_list = get_id2id(args.qrels_path)
    did2text = get_id2text(args.corpus_path)
    
    # 一些必要的操作
    if args.query_num != -1: q_embeds = q_embeds[:args.query_num]
    
    # 开始检索
    index = build_index(d_embeds, d_ids, args.index_flat_mode)
    D, I = index.search(q_embeds, args.topk)
    
    # 挑选并保存难负例
    if args.save_path:
        save_hard_neg(D, I, doc_list, id_d2q, args.save_path)
    
    # 计算本地召回率
    if args.local_recall_topk != -1:
        comput_local_recall(I, q_ids, id_q2d, 10)
        comput_local_recall(I, q_ids, id_q2d, 20)
        comput_local_recall(I, q_ids, id_q2d, 30)
        comput_local_recall(I, q_ids, id_q2d, 40)
        comput_local_recall(I, q_ids, id_q2d, 50)
        
    
"""
# 获取train_query到doc的映射（编号）
id_d2q = {}
doc_list = []
qrels_path = "/mnt/search/recall2/WenTianSearch/data/qrels.train.tsv"
reader = csv.reader(open(qrels_path), delimiter='\t')
for ind, line in enumerate(reader):
    if ind < 3:
        print(line)
    id_q = int(line[0])
    id_d = int(line[2])
    id_d2q[id_d] = id_q
    doc_list.append(id_d)


qrels_path = "/mnt/search/recall2/WenTianSearch/external_data/qrels.train.tsv"
reader = csv.reader(open(qrels_path), delimiter='\t')
for ind, line in enumerate(reader):
    if ind < 3:
        print(line)
    id_q = int(line[0]) + 100000
    id_d = int(line[2]) + 1001500
    id_q2d[id_q] = id_d
"""



"""
# 获取doc_id到文字的映射
did2text = {}
rels_path = "/mnt/search/recall2/WenTianSearch/data/corpus.tsv"
reader = csv.reader(open(rels_path), delimiter='\t')
        

for ind, line in enumerate(reader):
    # print(line, type(line))
    if ind < 3:
        print(line)
    did = int(line[0])
    text = line[1]
    did2text[did] = text


rels_path = "/mnt/search/recall2/WenTianSearch/external_data/corpus.tsv"
reader = csv.reader(open(rels_path), delimiter='\t')
        

for ind, line in enumerate(reader):
    # print(line, type(line))
    if ind < 3:
        print(line)
    did = int(line[0]) + 1001500
    text = line[1]
    did2text[did] = text


 # 传入特征维度
dim = 128

# IndexFlatIP表示利用内积来比较特征的相似度
# IndexFlatL2表示利用L2距离来比较特征的相似度
# 建立索引
index = faiss.IndexFlatIP(dim)
# index = faiss.IndexFlatL2(dim)
index2 = faiss.IndexIDMap(index)
index2.add_with_ids(d_embeds, d_ids)



# 开始检索
topk = 101
D, I = index.search(q_embeds[:], topk) # actual search
print(I.shape)
print(D.shape)
print(len(q_ids))


doc2simdoc = {}
# 开始保存
for i in range(len(D)):
    # 获取原始query
    simdocs = I[i]
    probs = D[i]
    # doc = simdocs[0]
    temp = doc_list[i]
    doc = id_d2q[temp]
    
    # 去掉原始query
    simdocs = I[i][1:]
    probs = D[i][1:]
    
    # 找候选项
    candidates = []
    for simdoc, prob in zip(simdocs, probs):
        # simdoc = did2text[simdoc+1]
        if prob <= 0.7:
            candidates.append(simdoc+1)
        # doc2simdoc[doc].append((str(simdoc), float(prob)))
    if len(candidates) == 0:
        candidates.append(simdocs[-1]+1)
    
    # 随机选一个
    shuffle(candidates)
    # if doc2simdoc.get(str(doc+1)) is not None:
    #    print(i)
    # print(str(doc), candidates)
    doc2simdoc[str(doc)] = str(candidates[0])
    
with open("./data/hard_neg_top100_70.json", "w") as f:
    json.dump(doc2simdoc, f)
"""  

# for i in I[0]:
#     print(i, did2text[i+1])
    # print(tokenizer.decode([int(i) for i in d_id2tokens[i+1]]))



# def comput_recall_k(k=10):
#     recall_k = 0
#     data_pairs = []
#     for i in range(len(I)):
#         query_id = q_ids[i]
#         recall_id = I[i]
#         recall_id = [d+1 for d in recall_id][:k]
#         label_id = id_q2d[query_id]

#         for d in recall_id:
#             if d == label_id:
#                 recall_k += 1
#     recall_k /= len(I)
    
#     with open("comput_recall_k_result.txt", "a+") as f:
#         f.write(str(k) + "\t" + str(recall_k) + "\n")



# with open("comput_recall_k_result.txt", "a+") as f:
#     patten='%Y%m%d_%H_%M_%S'
#     cur_time = time.strftime(patten, time.localtime(time.time()))
#     f.write(str(cur_time) + "\n\n\n")
                