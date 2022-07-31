



cd ..
pip install -r requirements.txt

# 创建保存排序模型的的文件夹
output_dir="./output"
mkdir $output
mkdir $output/epoch
mkdir $output/best_loss
mkdir $output/best_mrr
mkdir $output/last
mkdir $output/best_model


time="recall_1"
rm -rf ../../recall/data4rank/$time/.ipynb_checkpoints
# 训练排序模型
python train.py \
    --bert_model_path ../bert_pretrain/model4bert/chinese_L-12_H-768_A-12/ \
    --bert_config_path ../bert_pretrain/model4bert/chinese_L-12_H-768_A-12/bert_config.json \
    --bert_vocab_path ../bert_pretrain/model4bert/chinese_L-12_H-768_A-12/vocab.txt \
    --output_dir $output \
    --learning_rate 3e-5 \
    --batch_size 64 \
    --num_epochs 3 \
    --warmup_proportion 0.01 \
    --max_seq_length 128 \
    --save_steps 5000 \
    --eval_steps 50000 \
    --query_ids_file_path ../../recall/data4rank/train.query.json \
    --trainset_dir ../../recall/data4rank/$time \
    --corpus_ids_file_path ../../recall/data4rank/corpus.json \
    --dev_ids_file_path ./data_dev_ids


# 提交第二个epoch结束后保存的结果
cp $output/epoch/epoch1_models-100.* $output/best_model/
