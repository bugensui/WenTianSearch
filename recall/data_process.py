"""
1. 为faiss检索难负例做准备
2. 官方数据集和外部数据集处理后的结果，各自存放在各自的文件夹中
3. test必须为0
"""

import csv
import json
from random import shuffle
from tqdm import tqdm
import argparse

# 外部包
from utils import set_seeds, read_corpus, text_preprocess

def read_hard_neg(hard_neg=None):
    """
    读取难负例的映射表
    return doc2hard_neg: {query_id: hard_neg_id}
    """
    with open(hard_neg, "r") as f:
        doc2hard_neg = json.load(f)
    return doc2hard_neg

def data_write(lines, query_dict, corpus_dict, doc2hard_neg, writer, is_test=False, external_data=False):
    """
    将query和doc以<q,d>对儿的形式写入
    """
    ids = []
    max_len = 0
    for line in tqdm(lines):
        q_id = int(line[0])
        ids.append(q_id)
        v_id = int(line[2])
        q = query_dict[q_id]
        v = corpus_dict[v_id]
        q = text_preprocess(q)
        v = text_preprocess(v)
        if len(v) >= 108:
            print(v)
            continue
        if doc2hard_neg is not None:
            temp = corpus_dict[int(doc2hard_neg[str(q_id)])]
            temp = text_preprocess(temp)[:90]
            writer.writerow([q, v, temp])
        else:
            writer.writerow([q, v])
        max_len = max(len(q), max_len)
        max_len = max(len(v), max_len)
    print(max_len)
    if external_data: ids = [i+100000 for i in ids]
    if is_test:
        with open("test_ids", "a") as f:
            f.write(str(ids))
        

def make_qrels(query_dict, 
               corpus_dict,
               qrels_file='./data/qrels.train.tsv',
               writer_file='./data/query_doc.csv',
               test_file='./data/query_doc_test.csv',
               test_num=1000,
               hard_neg_path=None,
               data_shuffle=False,
               write_mode=None,
               external_data=False
               ):
    """
    :description :获取<query,doc>对
    :param query_dict: query中id到文本的映射
    :param corpus_dict: corpus中id到文本的映射
    :param qrels_file: query到点击对的[id]映射
    :param writer_file: 训练集的写入路径
    :param test_file: 雅正集的写入路径
    :param test_num: 验证集大小
    :param hard_neg_path: 难负例路径，None表示不包括难负例
    :param data_shuffle: 是否需要在拆分数据集之前，对总数据进行打乱顺序
    :param write_mode: 写入模式，是覆盖写还是追加写
    """
    # 必要准备
    click_doc2hard_neg_doc = read_hard_neg(hard_neg_path) if hard_neg_path else None
    w_m = write_mode
    assert w_m in ["a", "w"]
        
    # 读取总数据
    reader = csv.reader(open(qrels_file), delimiter='\t')
    writer = csv.writer(open(writer_file, w_m))
    test_writer = csv.writer(open(test_file, w_m))
    reader = [line for line in reader]
    
    # 是否需要写开头
    if w_m == "w":
        if hard_neg_path:
            writer.writerow(['query', 'doc', 'hard_neg_doc'])
            test_writer.writerow(['query', 'doc', 'hard_neg_doc'])
        else:
            writer.writerow(['query', 'doc'])
            test_writer.writerow(['query', 'doc'])
    
    # 切分数据
    if data_shuffle: shuffle(reader)    
    train_lines = reader[test_num:]
    test_lines = reader[:test_num]
    print(len(train_lines),len(test_lines))
        
    # 写入数据
    data_write(train_lines, query_dict, corpus_dict, click_doc2hard_neg_doc, writer)
    data_write(test_lines, query_dict, corpus_dict, click_doc2hard_neg_doc, test_writer, is_test=True, external_data=external_data)
    
    
if __name__ == '__main__':
    # 种随机种子
    set_seeds(2022)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gf_corpus_path", type=str, default='./data/corpus.tsv', help="官方cropus的路径") 
    parser.add_argument("--gf_train_query_path", type=str, default='./data/train.query.txt', help="官方train_query的路径") 
    parser.add_argument("--gf_qrels_file_path", type=str, default='./data/qrels.train.tsv', help="官方qrels.train.tsv的路径") 
    parser.add_argument("--gf_writer_file", type=str, default='./data/query_doc.csv', help="官方训练集的写入路径") 
    parser.add_argument("--gf_test_file", type=str, default='./data/query_doc_test.csv', help="官方验证集的写入路径") 
    parser.add_argument("--gf_hard_neg_path", type=str, default=None, help="官方数据集难负例的路径") 
    parser.add_argument("--gf_write_mode", type=str, default="w", choices=["a", "w"], help="官方数据集的写入方式") 
                        
    parser.add_argument("--ex_corpus_path", type=str, default='./external_data/corpus.tsv', help="外部cropus的路径") 
    parser.add_argument("--ex_train_query_path", type=str, default='./external_data/train.query.txt', help="外部train_query的路径") 
    parser.add_argument("--ex_qrels_file_path", type=str, default='./external_data/qrels.train.tsv', help="外部qrels.train.tsv的路径") 
    parser.add_argument("--ex_writer_file", type=str, default='./external_data/query_doc.csv', help="外部训练集的写入路径") 
    parser.add_argument("--ex_test_file", type=str, default='./external_data/query_doc_test.csv', help="外部验证集的写入路径") 
    parser.add_argument("--ex_hard_neg_path", type=str, default=None, help="外部数据集难负例的路径") 
    parser.add_argument("--ex_write_mode", type=str, default="w", choices=["a", "w"], help="外部数据集的写入方式") 
                        
    parser.add_argument("--test_num", type=int,  default=1000, required=True, help="验证集数目，必须显示指定") 
    parser.add_argument("--data_shuffle", action='store_true', default=False, help="是否需要在拆分数据集之前，对总数据进行打乱顺序") 
                        
    args = parser.parse_args()
    
    
    
    # 比赛官方数据
    corpus_dict = read_corpus(args.gf_corpus_path)
    query_dict = read_corpus(args.gf_train_query_path)
    make_qrels(query_dict,
               corpus_dict,
               qrels_file=args.gf_qrels_file_path,
               writer_file=args.gf_writer_file,
               test_file=args.gf_test_file,
               test_num=args.test_num,
               hard_neg_path=args.gf_hard_neg_path,
               data_shuffle=args.data_shuffle,
               write_mode=args.gf_write_mode
              )
    
    # 外部数据
    corpus_dict = read_corpus(args.ex_corpus_path)
    query_dict = read_corpus(args.ex_train_query_path)
    make_qrels(query_dict,
               corpus_dict,
               qrels_file=args.ex_qrels_file_path,
               writer_file=args.ex_writer_file,
               test_file=args.ex_test_file,
               test_num=args.test_num,
               hard_neg_path=args.ex_hard_neg_path,
               data_shuffle=args.data_shuffle,
               write_mode=args.ex_write_mode,
               external_data=True,
              )