# “阿里灵杰”问天引擎电商搜索算法赛
![image](./image/164611999657774301646119988943.png)
---

## 写在前面
队伍名：吨吨

复赛排名：13/2771

复赛成绩：0.3607

## 代码结构
以下是本项目主要代码结构及说明：

```
search/
├── recall # 召回模型
    ├── data # 官方数据
        ├── hard_neg # 基于官方数据的难负例
        ├── ... # 官方数据
    ├── external_data # 外部数据
        ├── hard_neg # 基于外部数据的难负例
        ├── ... # 外部数据
    ├── mlm_pretrain # MLM预训练
        ├── data4pretrain # 用于预训练的数据
        ├── model4pretrain # 用于预训练的模型权重
        ├── pretrain_data_process.py # 准备预训练用到的数据集
        ├── run_mlm.py # 进行mlm预训练
    ├── embedding_file # 产出的embedding
        ├── embed4fusai # 产出的用于复赛提交的embedding
            ├── ...
    ├── scripts # 运行脚本
        ├── run_mlm.sh # 进行MLM预训练文件
        ├── run_recall_model_step0.sh # 训练召回模型(无难负例)文件
        ├── run_hard_neg_data.sh # 基于前一版模型的embedding计算难负例文件
        ├── run_recall_model_step1.sh # 训练召回模型(有难负例)文件
        ├── run_prepare_rank_data.sh # 利用sota模型的embedding计算排序模型的训练数据文件
        ├── run_recall_model_step2.sh # 本地计算在训练集上的MRR指标文件
        ├── predict.sh # 预测脚本
    ├── modules # 一些模块（最终未使用）
        ├── xbm.py # 跨batch对比学习模块
    ├── simcse # simcse模型
        ├── models.py 
        ├── trainers.py
    ├── vocabulary # 词典
        ├── query.txt # query纠错&语义拓展词典
        ├── stopword.txt # 停用词典
    ├── data4rank # 存放训练排序模型的数据
        ├── ...
    ├── sota_model # 存放最好成绩对应的模型权重
        ├── ...
    ├── data_process.py # 数据预处理
    ├── train.py # 召回模型的训练
    ├── get_embedding_fusai.py  # 计算用于复赛提交的embedding
    ├── get_rank_data_2.py # 获取排序模型数据-总语料库和id映射信息
    ├── get_data_embedding.py # 获取指定数据的embedding信息
    ├── get_recall_top4rank_model.py # 获取排序模型数据-正负样本对
    ├── get_recall_top4hard_neg.py # 获取难负例
    ├── utils.py # 所有数据预处理方法及其他函数
|—— rank # 排序模型
    ├── scripts # 运行脚本
        ├── run_rank_model.sh # 排序模型的训练
        ├── predict.sh # 打包脚本
    |—— bert_pretrain4 # 基于tf1.x的预训练脚本 （tf1.x版预训练代码未使用）
        ├── ...
    ├── code # 代码（基于官方提供复赛demo修改）
        ├── bert # bert源代码
            ├── ...
        ├── data.py # 用于数据加载的文件
        ├── rank_model.py  # 排序模型
        ├── trainer.py.py  # 用于训练排序模型
        ├── wrapper.py # 用于打包排序模型
        ├── data_dev_ids # 训练召回模型时的验证集id，不参与排序模型的训练过程
        └── utils.py 工具包
```

# 数据与模型下载

- 比赛官方数据集: 
  - https://tianchi.aliyun.com/competition/entrance/531946/information 
  - 下载后放在 **search/recall/data/** 目录下
- 外部数据集
  - https://github.com/Alibaba-NLP/Multi-CPR/tree/main/data/ecom
  - 下载后放在 **search/recall/external_data/** 目录下
- hfl/chinese-roberta-wwm-ext： 
  - https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main
  - 下载后放在 **search/recall/mlm_pretrain/model4pretrain/hfl/chinese-roberta-wwm-ext/** 目录下
- BERT-Base-Chinese： 
  - https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
  - 下载后（需解压）放在 **search/rank/bert_pretrain/model4bert/chinese_L-12_H-768_A-12/** 目录下




# 召回模型

## Environment Requirements
The recall model has been trained on
 - Python3.7
 - CUDA 11.1
 - cuDNN 8.0.5
 - Pytorch 1.8.1
 - Ubuntu 18.04
 - NVCC11.1
 - NVIDIA RTX A4000 x 8

## 执行步骤

首先在根目录下切换到执行目录，并且安装相关环境：

```
cd ./recall 
pip install -r requirements.txt
apt-get install libopenblas-dev # 安装faiss时需要的环境
apt-get install libomp-dev # 安装faiss时需要的环境
cd ./scripts
```

召回模型包含以下六步（**执行前请先查看注意事项**）
 - Step 1: 在电商语料下进行Domain Adaptive Pretraining（3.5h）
 - Step 2: 基于SimCSE训练召回模型(无难负例)（0.4h）
 - Step 3: 基于Step2模型计算的embedding产出难负例文件（0.3h）
 - Step 4: 基于SimCSE训练召回模型(有难负例)（0.6h）
 - Step 5: 利用Step4模型在测试集上预测结果（0.3h）
 - Step 6: 利用Step4模型计算的embedding产出排序模型的训练数据文件（0.5h）


#### Step 1: 在电商语料下进行Domain Adaptive Pretraining
```
sh run_mlm.sh
```

#### Step 2: 基于SimCSE训练召回模型(无难负例)
```
sh run_recall_model_step0.sh
```

#### Step 3: 基于Step2模型计算的embedding产出难负例文件
```
sh run_hard_neg_data.sh
```

#### Step 4: 基于SimCSE训练召回模型(有难负例)
```
sh run_recall_model_step1.sh
```

#### Step 5: 利用Step4模型在测试集上预测结果
```
sh predict.sh
```

#### Step 6: 利用Step4模型计算的embedding产出排序模型的训练数据文件
```
sh run_prepare_rank_data.sh
```


## 注意事项
 - Step1-Step5 需要在 **recall/scripts/** 目录下执行
 - Step2和Step4内部执行操作基本一致，都是**准备数据-训练-预测embedding**，有时会出现准备数据完成后，训练部分无法启动，这时候需要**手动注释掉准备数据部分的shell代码，重新执行Step2或Step4**。
 - recall_0表示step2训练的模型，这个模型只是中间产物。 最终的召回模型是recall_1对应的模型
 - 在pip install faiss安装后，还需要执行下面两条指令才能正常使用
    ```
    apt-get install libopenblas-dev
    apt-get install libomp-dev
    ```



---
# 排序模型

## Environment Requirements
The rank model has been trained on
 - Python3.6
 - Cudnn7.6.4
 - Cupy-cuda100:7.0.0
 - Keras:2.2.4
 - Tensorflow-gpu:1.12.0
 - Ubuntu 18.04
 - NVIDIA Tesla P100-16GB x 1


## 执行步骤
首先在根目录下切换到执行目录，并且安装相关环境：

```
cd ./rank/code
pip install -r requirements.txt
cd ./scripts
```

召回模型包含以下两步（**执行前请先查看注意事项**）
 - Step 1: 排序模型的训练（10-12h）
 - Step 2: 排序模型的打包（0.01h）

#### Step 1: 排序模型的训练
```
sh run_rank_model.sh
```

#### Step 2: 排序模型的打包
```
sh predict.sh
```

## 注意事项
 - Step1 需要在 **rerank/code/scripts/** 目录下执行
 - 训练排序模型前要删除 **recall/data4rank/recall_1/**  下的 **.ipynb_checkpoints** 文件夹
---

# 打包提交
## embedding地址
路径会在执行脚本后自动创建
```
recall/embedding_file/embed4fusai/recall_1/
```

## 排序模型地址
路径会在执行脚本后自动创建
```
rank/code/output/best_model/wrapper
```

## rerank_size为默认值10
