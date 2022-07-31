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
from utils import text_preprocess, set_seeds, query_vacab

device = "cuda" if torch.cuda.is_available() else "cpu"
    
batch_size = 100
use_pinyin = False


def encode_fun(texts, model, query_vacab=None):
    if query_vacab is not None:
        new_texts = []
        for text in texts:
            if query_vacab.get(text) is not None:
                new_texts.append(query_vacab[text])
                print(text, query_vacab[text])
            else:
                new_texts.append(text)
        texts = new_texts

    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=115)
    inputs.to(device)
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
        embeddings = embeddings.squeeze(0).cpu().numpy()
    return embeddings, inputs['input_ids']




if __name__ == '__main__':
    set_seeds(2022)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./sota_model/0513/', help="召回模型路径") 
    parser.add_argument("--corpus_path", type=str, default='./data/corpus.tsv', help="cropus的路径") 
    parser.add_argument("--dev_query_path", type=str, default='./sota_model/0423/data/dev.query.txt', help="测试集querys的路径") 
    parser.add_argument("--write_query_embed_file", type=str, default='./embedding_file/embed4fusai/query_embedding', help="querys的embed保存路径")
    parser.add_argument("--write_doc_embed_file", type=str, default='./embedding_file/embed4fusai/doc_embedding', help="docs的embed保存路径")
    args = parser.parse_args()
    print(args)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = BertForCL.from_pretrained(args.model_path)
    model.to(device)

    corpus = [line[1] for line in csv.reader(open(args.corpus_path), delimiter='\t')]
    query = [line[1] for line in csv.reader(open(args.dev_query_path), delimiter='\t')]

    query_vacab = query_vacab()
    
    query_embedding_file = csv.writer(open(args.write_query_embed_file, 'w'), delimiter='\t')
    
    for i in tqdm(range(0, len(query), batch_size)):
        batch_text = query[i:i + batch_size]
        batch_text = [text_preprocess(text) for text in batch_text]
        temp_embedding, input_ids = encode_fun(batch_text, model, query_vacab)
        assert len(temp_embedding) == len(input_ids)
        for j in range(len(temp_embedding)):
            input_id = input_ids[j].detach().cpu().numpy().tolist()
            # input_id = input_id[1:]  # 去cls
            new_input_id = []
            for id in input_id:
                if id == 0: break
                new_input_id.append(id)
            input_id = new_input_id   # 去 pad
            line = {"qid":str(i + j + 200001), "input_ids":input_id}
            
            input_id = [str(id) for id in input_id]
            input_id = ','.join(input_id)

            writer_str = temp_embedding[j].tolist()
            writer_str = [format(s, '.8f') for s in writer_str]
            writer_str = ','.join(writer_str)
            query_embedding_file.writerow([i + j + 200001, writer_str, input_id])
            
    doc_embedding_file = csv.writer(open(args.write_doc_embed_file, 'w'), delimiter='\t')
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_text = corpus[i:i + batch_size]
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
