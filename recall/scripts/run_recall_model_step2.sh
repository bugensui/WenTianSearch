# 注意需要修改的地方：
# 1. --model_path  
# 2. --topk
# 3. 第一步可以不执行，如果执行过计算难负例的话

cd ..
apt-get install libopenblas-dev
apt-get install libomp-dev

model_path="./sota_model/recall_1"
time=${model_path#*./sota_model/}
query_embedd_path=./embedding_file/$time/train_query_embedding
doc_embed_path=./embedding_file/$time/doc_embedding

# 生成数据的embedding，并写入制定文件中-官方
python get_data_embedding.py \
    --model_path $model_path \
    --corpus_path ./data/corpus.tsv \
    --querys_path ./data/train.query.txt \
    --write_query_embed_file $query_embedd_path \
    --write_doc_embed_file $doc_embed_path \
    --query_delimiter "t"

# # 计算在训练集上的召回率
python get_recall_top4hard_neg.py \
    --query_embed_path $query_embedd_path \
    --doc_embed_path $doc_embed_path  \
    --qrels_path ./data/qrels.train.tsv \
    --corpus_path ./data/corpus.tsv \
    --topk 50 \
    --query_num -1 \
    --local_recall_topk \
    --index_flat_mode l2

cd scripts