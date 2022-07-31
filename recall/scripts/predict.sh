# 训练召回模型(有难负例)

cd ..

# 创建保存embedding的文件夹
time="recall_1"
mkdir ./embedding_file/embed4fusai/$time



# 在测试集上生成embedding
python get_embedding_fusai.py \
    --model_path $save_model_path \
    --corpus_path ./data/corpus.tsv \
    --dev_query_path ./data/dev.query.txt \
    --write_query_embed_file ./embedding_file/embed4fusai/$time/query_embedding \
    --write_doc_embed_file ./embedding_file/embed4fusai/$time/doc_embedding
