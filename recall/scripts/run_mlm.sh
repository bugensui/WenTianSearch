
# 1. 准备预训练的数据
# 2. 进行mlm预训练

cd ..
pip install -r requirements.txt

cd ./mlm_pretrain

mlm_train_data_path="./data4pretrain/training_corpus.txt"
mlm_dev_data_path="./data4pretrain/validation_corpus.txt"
python pretrain_data_process.py \
    --gf_corpus_path ../data/corpus.tsv \
    --gf_train_query_path ../data/train.query.txt \
    --gf_dev_query_path ../data/dev.query.txt \
    --ex_corpus_path ../external_data/corpus.tsv \
    --ex_train_query_path ../external_data/train.query.txt \
    --ex_dev_query_path ../external_data/dev.query.txt \
    --mlm_train_data_writer_file $mlm_train_data_path \
    --mlm_dev_data_writer_file $mlm_dev_data_path \
    --dev_size 10000


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPU=8
PORT_ID=$(expr $RANDOM + 1000)
python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID  run_mlm.py \
    --model_name_or_path ./model4pretrain/hfl/chinese-roberta-wwm-ext/ \
    --num_train_epochs 10 \
    --train_file $mlm_train_data_path \
    --validation_file $mlm_dev_data_path \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 64 \
    --do_train \
    --do_eval \
    --output_dir ./tmp \
    --line_by_line \
    --eval_steps 500  \
    --logging_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --max_seq_length 128 \
    --save_total_limit 3 \
