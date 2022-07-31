import os
import csv
import sys
import json

import torch
import argparse
from tqdm import tqdm

sys.path.append("..")
from transformers import AutoTokenizer

# 外部包
from utils import set_seeds, text_preprocess

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
batch_size = 5000
use_pinyin = False


def encode_fun(texts):
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=115)
    inputs.to(device)
    return inputs['input_ids']


def get_rank_data(rank_corpus, rank_train_query, corpus, query, is_external=False):
    len_train_query, len_corpus = 0, 0
    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        batch_text = [text_preprocess(text) for text in batch_text]
        input_ids = encode_fun(batch_text)
        for j in range(len(input_ids)):
            input_id = input_ids[j].detach().cpu().numpy().tolist()
            input_id = input_id[1:]  # 去cls
            new_input_id = []
            for id in input_id:
                if id == 0: break
                new_input_id.append(id)
            input_id = new_input_id   # 去 pad
            if is_external:
                line = {"qid":str((i + j + 100001)), "input_ids":input_id}
            else:
                line = {"qid":str(i + j + 1), "input_ids":input_id}
            with open(rank_train_query, "a+") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
               
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_text = corpus[i:i + batch_size]
        batch_text = [text_preprocess(text) for text in batch_text]
        input_ids = encode_fun(batch_text)
        for j in range(len(input_ids)):
            input_id = input_ids[j].detach().cpu().numpy().tolist()
            input_id = input_id[1:]  # 去cls
            new_input_id = []
            for id in input_id:
                if id == 0: break
                new_input_id.append(id)
            input_id = new_input_id   # 去 pad
            if is_external:
                line = {"qid":str((i + j + 1001501)), "input_ids":input_id}
            else:
                line = {"qid":str(i + j + 1), "input_ids":input_id}
            with open(rank_corpus, "a+") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    set_seeds(2022)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gf_corpus_path", type=str, default='./data/corpus.tsv', help="cropus的路径") 
    parser.add_argument("--gf_train_querys_path", type=str, default='./data/train.query.txt', help="train_querys的路径") 
    
    parser.add_argument("--ex_corpus_path", type=str, default='./external_data/corpus.tsv', help="外部cropus的路径") 
    parser.add_argument("--ex_train_querys_path", type=str, default='./external_data/train.query.txt', help="外部train_querys的路径") 
    
    parser.add_argument("--write_rank_corpus_file", type=str, default='./data4rank/corpus.json', help="排序模型的语料")
    parser.add_argument("--write_rank_train_query_file", type=str, default='./data4rank/train.query.json', help="训练query-tokenid")
    args = parser.parse_args()
    print(args)
    
    
    rank_corpus = args.write_rank_corpus_file
    rank_train_query = args.write_rank_train_query_file
    if os.path.exists(rank_train_query):
        os.remove(rank_train_query)
    if os.path.exists(rank_corpus):
        os.remove(rank_corpus)

    
    corpus = [line[1] for line in csv.reader(open(args.gf_corpus_path), delimiter='\t')]
    query = [line[1] for line in csv.reader(open(args.gf_train_querys_path), delimiter='\t')]
    print(len(corpus), len(query))
    get_rank_data(rank_corpus, rank_train_query, corpus, query)
    
    
    external_corpus = [line[1] for line in csv.reader(open(args.ex_corpus_path), delimiter='\t')]
    external_query = [line[1] for line in csv.reader(open(args.ex_train_querys_path), delimiter='\t')]
    print(len(corpus), len(query))
    
    get_rank_data(rank_corpus, rank_train_query, external_corpus, external_query, is_external=True)
    