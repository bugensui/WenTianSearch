# 注意需要修改的地方：
# 1. time 

cd ..

# 准备总的数据集（id到token的映射）
python get_rank_data_2.py \
    --gf_corpus_path ./data/corpus.tsv \
    --gf_train_querys_path ./data/train.query.txt \
    --ex_corpus_path ./external_data/corpus.tsv \
    --ex_train_querys_path ./external_data/train.query.txt \
    --write_rank_corpus_file ./data4rank/corpus.json \
    --write_rank_train_query_file ./data4rank/train.query.json

# 准备训练用的数据对-官方
time="recall_1"
mkdir ./data4rank/$time
save_path=./data4rank/$time/data_pairs
python get_recall_top4rank_model.py \
    --query_embed_path ./embedding_file/$time/train_query_embedding \
    --doc_embed_path ./embedding_file/$time/doc_embedding  \
    --qrels_path ./data/qrels.train.tsv \
    --corpus_path ./data/corpus.tsv \
    --topk 40 \
    --save_path $save_path \
    --index_flat_mode l2 \
    --write_model w \

# 准备训练用的数据对-外部
python get_recall_top4rank_model.py \
    --query_embed_path ./embedding_file/$time/external_train_query_embedding \
    --doc_embed_path ./embedding_file/$time/external_doc_embedding  \
    --qrels_path ./external_data/qrels.train.tsv \
    --corpus_path ./external_data/corpus.tsv \
    --topk 40 \
    --save_path $save_path \
    --index_flat_mode l2 \
    --write_model a \
    --external_data
    
cd scripts 
