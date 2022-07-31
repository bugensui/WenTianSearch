
cd ..

# 创建保存排序模型的的文件夹
output_dir="./output"


# 打包到指定文件夹
python wrapper.py \
    --bert_config_path ../bert_pretrain/model4bert/bert_config.json \
    --ckpt_to_convert $output/best_model/epoch1_models-100 \
    --output_dir $output/best_model/wrapper \
    --max_seq_length 128

