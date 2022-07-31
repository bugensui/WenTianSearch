mlm_train_data_path="/mnt/search/rank/demo/tianchi_ranker/bert_pretrain/data4pretrain/training_corpus.txt"
mlm_dev_data_path="/mnt/search/rank/demo/tianchi_ranker/bert_pretrain/data4pretrain/validation_corpus.txt"
BERT_BASE_DIR="/mnt/search/gen/simbert/model/chinese_L-12_H-768_A-12"

cd ..
python pretrain_data_process.py \
    --gf_corpus_path /mnt/search/recall2/WenTianSearch/data/corpus.tsv \
    --gf_train_query_path /mnt/search/recall2/WenTianSearch/data/train.query.txt \
    --gf_dev_query_path /mnt/search/recall2/WenTianSearch/data/dev.query.txt \
    --ex_corpus_path /mnt/search/recall2/WenTianSearch/external_data/corpus.tsv \
    --ex_train_query_path /mnt/search/recall2/WenTianSearch/external_data/train.query.txt \
    --ex_dev_query_path /mnt/search/recall2/WenTianSearch/external_data/dev.query.txt \
    --mlm_train_data_writer_file $mlm_train_data_path \
    --mlm_dev_data_writer_file $mlm_dev_data_path \
    --dev_size 0


python create_pretraining_data.py \
  --input_file=$mlm_train_data_path \
  --output_file=./tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=2022 \
  --dupe_factor=5
  
  
  
python run_pretraining.py \
  --input_file=/mnt/search/rank/demo/tianchi_ranker/bert_pretrain/bert/tmp/tf_examples.tfrecord \
  --output_dir=./tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=64 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=46000 \
  --num_warmup_steps=300 \
  --learning_rate=2e-5
  
  
cd ./tmp/pretraining_output
cp ./model.ckpt-46000.* ../best_model/
