import csv
import random
import argparse
from tqdm import tqdm

import sys
sys.path.append("../")
from utils import set_seeds, text_preprocess


def read_corpus(file_path=None):
    reader = csv.reader(open(file_path,encoding='utf-8'), delimiter='\t')
    total_list = []
    for line in tqdm(reader):
        corpus = line[1]
        corpus = text_preprocess(corpus)
        total_list.append(corpus)
    return total_list

if __name__ == '__main__':
    # 固定随机种子
    set_seeds(2022)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gf_corpus_path", type=str, default='../data/corpus.tsv', help="官方cropus的路径") 
    parser.add_argument("--gf_train_query_path", type=str, default='../data/train.query.txt', help="官方train_query的路径") 
    parser.add_argument("--gf_dev_query_path", type=str, default='../data/dev.query.txt', help="官方dev_query的路径") 
                        
    parser.add_argument("--ex_corpus_path", type=str, default='../external_data/corpus.tsv', help="外部cropus的路径") 
    parser.add_argument("--ex_train_query_path", type=str, default='../external_data/train.query.txt', help="外部train_query的路径") 
    parser.add_argument("--ex_dev_query_path", type=str, default='../external_data/dev.query.txt', help="外部dev_query的路径") 
    
    parser.add_argument("--mlm_train_data_writer_file", type=str, default=None, help="用于训练mlm的数据导入路径") 
    parser.add_argument("--mlm_dev_data_writer_file", type=str, default=None, help="用于验证mlm的数据导入路径") 
    
    parser.add_argument("--dev_size", type=int, default=10000, help="用于验证mlm的数据大小") 
    args = parser.parse_args()
    print(args)
    
    
    # 读取语料
    corpus_list = read_corpus(args.gf_corpus_path)
    train_query = read_corpus(args.gf_train_query_path)
    test_query = read_corpus(args.gf_dev_query_path)
    external_corpus_list = read_corpus(args.ex_corpus_path)
    external_train_query = read_corpus(args.ex_train_query_path)
    external_test_query = read_corpus(args.ex_dev_query_path)
    
    # 合并打乱
    all_corpus = corpus_list+train_query+test_query+external_corpus_list+external_train_query+external_test_query
    random.shuffle(all_corpus)
    
    # 分训练验证
    test_size = args.dev_size
    train = all_corpus[:-test_size]
    test = all_corpus[-test_size:]
    
    # 写入文件
    train_writer = csv.writer(open(args.mlm_train_data_writer_file, 'w',encoding='utf-8'))
    test_writer = csv.writer(open(args.mlm_dev_data_writer_file, 'w',encoding='utf-8'))
    for line in tqdm(train):
        train_writer.writerow([line])
    for line in tqdm(test):
        test_writer.writerow([line])