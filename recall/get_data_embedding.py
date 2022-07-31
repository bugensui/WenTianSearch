"""
1. 
获取官方数据的10Wquery和100wdoc的embedding
"""

import os
import csv
import sys
import json

import torch
import argparse
from tqdm import tqdm

sys.path.append("..")
from simcse.models import BertForCL
from transformers import AutoTokenizer

# 外部包
from utils import set_seeds, text_preprocess

device = "cuda" if torch.cuda.is_available() else "cpu"    
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
batch_size = 100

def encode_fun(texts, model):
    """将文本编码为embed
    """
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=115)
    inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
        embeddings = embeddings.squeeze(0).cpu().numpy()
    return embeddings, inputs['input_ids']

def build_model(args):
    """加载模型
    """
    model = BertForCL.from_pretrained(args.model_path)
    model.to(device)
    return model


def write_embed(lines, write_embed_file, delete_cls, delete_sep):
    """将embed信息写入指定文件中
    """
    embed_file = csv.writer(open(write_embed_file, 'w'), delimiter='\t')
    for i in tqdm(range(0, len(lines), batch_size)):
        batch_text = lines[i:i + batch_size]
        batch_text = [text_preprocess(text) for text in batch_text]
        temp_embedding, input_ids = encode_fun(batch_text, model)
        assert len(temp_embedding) == len(input_ids)
        for j in range(len(temp_embedding)):
            input_id = input_ids[j].detach().cpu().numpy().tolist()
            if delete_cls: input_id = input_id[1:]  # 去cls
            new_input_id = []
            for id in input_id:
                if id == 0: break
                new_input_id.append(id)
            input_id = new_input_id   # 去 pad
            if delete_sep: input_id = input_id[:-1]
            
            input_id = [str(id) for id in input_id]
            input_id = ','.join(input_id)

            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            embed_file.writerow([i + j + 1, writer_str, input_id])
    

if __name__ == '__main__':
    set_seeds(2022)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/mnt/search/recall2/WenTianSearch/sota_model/0513/', help="召回模型路径") 
    parser.add_argument("--corpus_path", type=str, default='./data/corpus.tsv', help="cropus的路径") 
    parser.add_argument("--querys_path", type=str, default='./data/train.query.txt', help="querys的路径") 
    parser.add_argument("--delete_head", action='store_true', default=False, help="是否删掉query里面第一行") 
    parser.add_argument("--query_delimiter", type=str, default="t", choices=["t", ","], help="获取query时的分隔符") 
    parser.add_argument("--write_query_embed_file", type=str, default='./embedding_file/train_query_embedding', help="querys的embed保存路径")
    parser.add_argument("--write_doc_embed_file", type=str, default='./embedding_file/doc_embedding', help="docs的embed保存路径")
    args = parser.parse_args()
    print(args)
    
    # 一些必要的处理
    if args.query_delimiter == "t": query_delimiter = "\t"

    # 加载模型
    model = build_model(args)
    
    # 加载数据
    corpus = [line[1] for line in csv.reader(open(args.corpus_path), delimiter='\t')]
    query = [line[1] for line in csv.reader(open(args.querys_path), delimiter=query_delimiter)]
    if args.delete_head: query = query[1:]  # 一般txt格式用[\t]且不用删开头， csv格式用[,]且需要删开头
        
    # 写入embed
    write_embed(query, args.write_query_embed_file, delete_cls=False, delete_sep=False)
    write_embed(corpus, args.write_doc_embed_file, delete_cls=True, delete_sep=True)
    
    
    
    """
    query_embedding_file = csv.writer(open('/mnt/search/recall2/WenTianSearch/train_query_embedding', 'w'), delimiter='\t')

    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        batch_text = [text_preprocess(text) for text in batch_text]
        temp_embedding, input_ids = encode_fun(batch_text, model)
        assert len(temp_embedding) == len(input_ids)
        for j in range(len(temp_embedding)):
            input_id = input_ids[j].detach().cpu().numpy().tolist()
            # input_id = input_id[1:]  # 去cls
            new_input_id = []
            for id in input_id:
                if id == 0: break
                new_input_id.append(id)
            input_id = new_input_id   # 去 pad
            line = {"qid":str(i + j + 1), "input_ids":input_id}
            
            input_id = [str(id) for id in input_id]
            input_id = ','.join(input_id)

            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            query_embedding_file.writerow([i + j + 1, writer_str, input_id])
    
    doc_embedding_file = csv.writer(open('/mnt/search/recall2/WenTianSearch/doc_embedding', 'w'), delimiter='\t')
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_text = corpus[i:i + batch_size]
        batch_text = [text_preprocess(text) for text in batch_text]
        temp_embedding, input_ids = encode_fun(batch_text, model)
        for j in range(len(temp_embedding)):
            input_id = input_ids[j].detach().cpu().numpy().tolist()
            input_id = input_id[1:]  # 去cls
            new_input_id = []
            for id in input_id:
                if id == 0: break
                new_input_id.append(id)
            input_id = new_input_id[:-1]   # 去 pad 和 sep
            line = {"qid":str(i + j + 1), "input_ids":input_id}
            
            input_id = [str(id) for id in input_id]
            input_id = ','.join(input_id)

            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            doc_embedding_file.writerow([i + j + 1, writer_str, input_id])
    """