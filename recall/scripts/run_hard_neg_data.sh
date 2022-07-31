# 注意需要修改的地方：
# 1. --model_path
# 2. 注释时候，model_path 和 time 一定不能注释

# 获取总的数据集
cd ..
python data_process.py \
    --gf_corpus_path ./data/corpus.tsv \
    --gf_train_query_path ./data/train.query.txt \
    --gf_qrels_file_path ./data/qrels.train.tsv \
    --gf_writer_file ./data/query_doc.csv \
    --gf_test_file ./data/query_doc_test.csv \
    --gf_write_mode w \
    --ex_corpus_path ./external_data/corpus.tsv \
    --ex_train_query_path ./external_data/train.query.txt \
    --ex_qrels_file_path ./external_data/qrels.train.tsv \
    --ex_writer_file ./external_data/query_doc.csv \
    --ex_test_file ./external_data/query_doc_test.csv \
    --ex_write_mode w \
    --test_num 0
    
    
# # 生成数据的embedding，并写入制定文件中-官方
model_path="./sota_model/recall_0"
time=${model_path#*./sota_model/}
mkdir ./embedding_file/$time
python get_data_embedding.py \
    --model_path $model_path \
    --corpus_path ./data/corpus.tsv \
    --querys_path ./data/train.query.txt \
    --write_query_embed_file ./embedding_file/$time/train_query_embedding \
    --write_doc_embed_file ./embedding_file/$time/doc_embedding \
    --query_delimiter "t"

# 生成数据的embedding，并写入制定文件中-外部
python get_data_embedding.py \
    --model_path $model_path \
    --corpus_path ./external_data/corpus.tsv \
    --querys_path ./external_data/train.query.txt \
    --write_query_embed_file ./embedding_file/$time/external_train_query_embedding \
    --write_doc_embed_file ./embedding_file/$time/external_doc_embedding \
    --query_delimiter "t"



# 生成难负例-官方
apt-get install libopenblas-dev
apt-get install libomp-dev

mkdir ./data/hard_neg/$time
mkdir ./external_data/hard_neg/$time
python get_recall_top4hard_neg.py \
    --query_embed_path ./embedding_file/$time/train_query_embedding \
    --doc_embed_path ./embedding_file/$time/doc_embedding  \
    --qrels_path ./data/qrels.train.tsv \
    --corpus_path ./data/corpus.tsv \
    --topk 101 \
    --query_num -1 \
    --save_path ./data/hard_neg/$time/hard_neg_top100_70.json \
    --index_flat_mode ip
    
# 生成难负例-外部
python get_recall_top4hard_neg.py \
    --query_embed_path ./embedding_file/$time/external_train_query_embedding \
    --doc_embed_path ./embedding_file/$time/external_doc_embedding \
    --qrels_path ./external_data/qrels.train.tsv \
    --corpus_path ./external_data/corpus.tsv \
    --topk 101 \
    --query_num -1 \
    --save_path ./external_data/hard_neg/$time/hard_neg_top100_70.json \
    --index_flat_mode ip
    
    
cd scripts