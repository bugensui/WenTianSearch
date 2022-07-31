# 训练召回模型(无难负例)

cd ..

# 获取总的数据集
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
     --ex_writer_file ./data/query_doc.csv \
     --ex_test_file ./data/query_doc_test.csv \
     --ex_write_mode a \
     --test_num 1000 \
     --data_shuffle \
    
    
# 训练召回模型
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)
export OMP_NUM_THREADS=8
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID  train.py \
    --model_name_or_path ./mlm_pretrain/tmp \
    --train_file "./data/query_doc.csv" \
    --validation_file "./data/query_doc_test.csv" \
    --output_dir ./result/unsup-simcse \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --metric_for_best_model eval_acc \
    --learning_rate 3e-5 \
    --max_seq_length 108 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 10 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --do_fgm




# 删除检查点-并保存模型
time="recall_0"
rm -rf ./result/unsup-simcse/checkpoi*
save_model_path=./sota_model/$time
mkdir $save_model_path
mv ./result/unsup-simcse/* $save_model_path



# 创建保存embedding的文件夹
mkdir ./embedding_file/embed4fusai/$time




# 在测试集上生成embedding
python get_embedding_fusai.py \
    --model_path $save_model_path \
    --corpus_path ./data/corpus.tsv \
    --dev_query_path ./data/dev.query.txt \
    --write_query_embed_file ./embedding_file/embed4fusai/$time/query_embedding \
    --write_doc_embed_file ./embedding_file/embed4fusai/$time/doc_embedding




# 打包
cd ./embedding_file/embed4fusai/$time
tar zcvf foo.tar.gz doc_embedding query_embedding
cd ../../


cd scripts
