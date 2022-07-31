import os
import json
import argparse
import tensorflow as tf
import time 
import random
import numpy as np
from datetime import datetime

from data import DataProcessor
from bert import modeling
from rank_model import Ranker

from tqdm import tqdm

# pip install gast==0.2.2
     
class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.bert_config_path = args.bert_config_path
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.warmup_proportion = args.warmup_proportion
        self.max_seq_length = args.max_seq_length
        self.learning_rate = args.learning_rate
        self.bert_model_path = args.bert_model_path
        self.processor = DataProcessor(args.corpus_ids_file_path, args.bert_vocab_path)
        self.dev_ids = self.get_dev_ids()

    def create_model(self):
        """创建模型"""
        model = Ranker(self.bert_config_path, is_training=self.args.is_training, num_train_steps=self.num_train_steps, num_warmup_steps=self.num_warmup_steps, learning_rate=self.learning_rate)
        return model

    def get_dev_ids(self):
        with open(self.args.dev_ids_file_path, "r") as f:
            dev_ids = eval(f.read())
            dev_ids = [str(i) for i in dev_ids]
        return dev_ids

    def train(self):
        """开始训练"""
        train_examples = self.processor.get_train_examples(self.args.query_ids_file_path, self.args.trainset_dir, self.dev_ids, "train")
        eval_examples_lst = self.processor.get_dev_examples(self.args.query_ids_file_path, self.args.trainset_dir, self.dev_ids, "dev")
        
        self.num_train_steps = int(len(train_examples) / self.batch_size * self.num_epochs)
        self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion) 
    
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        tf.logging.info("***** Running training *****")
        tf.logging.info(" Num examples = %d", len(train_examples))
        tf.logging.info(" Batch size = %d", self.batch_size)
        tf.logging.info(" Num steps = %d", self.num_train_steps)
        
        num_batches = len(train_examples) // self.batch_size
        self.model = self.create_model()

        with tf.Session() as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, self.bert_model_path)
            tf.train.init_from_checkpoint(self.bert_model_path, assignment_map) 
            tf.logging.info("***** Trainable Variables *****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                #tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape, init_string)
            sess.run(tf.variables_initializer(tf.global_variables()))

            best_loss = 100
            best_mrr = 0
            starttime = time.time()
            for i in tqdm(range(self.num_epochs)):
                current_step = 0
                train_losses, eval_losses, eval_mrrs = [], [], []
                loss = 0
                print("***** epoch-{} *******".format(i))
                if i > 0: # 每一个epoch都随机化负例
                    tf.logging.info("**** get train_examples ****")
                    train_examples = self.processor.get_train_examples(self.args.query_ids_file_path, self.args.trainset_dir, self.dev_ids, "train")
                    random.shuffle(train_examples) 
                features = self.processor.get_features(train_examples, max_seq_length=self.max_seq_length)
                input_ids_lst, input_mask_lst, token_type_ids_lst, label_ids_lst = self.processor.get_inputs(features)
                
                # eval_features = self.processor.get_features(eval_examples, max_seq_length=self.max_seq_length)
                # eval_input_ids_lst, eval_input_mask_lst, eval_token_type_ids_lst, eval_label_ids_lst = self.processor.get_inputs(eval_features)
                
                for j in tqdm(range(num_batches)):
                    start = j * self.batch_size
                    end = start + self.batch_size
                    batch_features = {"input_ids": input_ids_lst[start: end], 
                                      "input_mask": input_mask_lst[start: end], 
                                      "token_type_ids": token_type_ids_lst[start: end], 
                                      "label_ids": label_ids_lst[start: end]}
                    
                    temp_train_loss, logits, prob = self.model.train(sess, batch_features)
                    train_losses.append(temp_train_loss)
                    loss += temp_train_loss
                    
                    # eval
                    if current_step > 0 and current_step % self.args.eval_steps == 0:
                        eval_loss = 0
                        eval_count = 0
                        eval_mrr = 0
                        for eval_examples in tqdm(eval_examples_lst):
                            eval_features = self.processor.get_features(eval_examples, max_seq_length=self.max_seq_length)
                            eval_input_ids_lst, eval_input_mask_lst, eval_token_type_ids_lst, eval_label_ids_lst = self.processor.get_inputs(eval_features)

                            eval_batch_features = {"input_ids": eval_input_ids_lst, 
                                                "input_mask": eval_input_mask_lst, 
                                                "token_type_ids": eval_token_type_ids_lst, 
                                                "label_ids": eval_label_ids_lst}
                            temp_eval_loss, logits, prob, temp_eval_mrr = self.model.eval(sess, eval_batch_features)
                            
                            eval_loss += temp_eval_loss
                            eval_mrr += temp_eval_mrr
                        eval_loss /= len(eval_examples_lst)
                        eval_mrr /= len(eval_examples_lst)
                        eval_losses.append(eval_loss)
                        eval_mrrs.append(eval_mrr)
                        """
                        for eval_start in tqdm(range(0, len(eval_examples), self.batch_size)):
                            eval_count += 1
                            eval_end = eval_start + self.batch_size
                            eval_batch_features = {"input_ids": eval_input_ids_lst[eval_start: eval_end], 
                                                "input_mask": eval_input_mask_lst[eval_start: eval_end], 
                                                "token_type_ids": eval_token_type_ids_lst[eval_start: eval_end], 
                                                "label_ids": eval_label_ids_lst[eval_start: eval_end]}
                            temp_loss, logits, prob = self.model.eval(sess, eval_batch_features)
                            eval_loss += temp_loss
                        eval_loss /= eval_count
                        """

                        if eval_loss < best_loss:
                            best_loss = eval_loss
                            # tf.logging.info("***** saving model to %s ****", self.args.output_dir)
                            ckpt_name_prefix = "best_loss/best_loss_models"
                            save_path = os.path.join(self.args.output_dir, ckpt_name_prefix)
                            self.model.saver.save(sess, save_path, global_step=100)
                        
                        if eval_mrr > best_mrr:
                            best_mrr = eval_mrr
                            # tf.logging.info("***** saving model to %s ****", self.args.output_dir)
                            ckpt_name_prefix = "best_mrr/best_mrr_models"
                            save_path = os.path.join(self.args.output_dir, ckpt_name_prefix)
                            self.model.saver.save(sess, save_path, global_step=100)
                        
                        
                        tf.logging.info("*****【%s】, training_step: %d 【%s】, train_loss: %f, eval_loss: %f, eval_mrr: %f", datetime.now().strftime("%H:%M:%S"), current_step, str(100*current_step/self.num_train_steps)+"%", loss/(j+1), eval_loss, eval_mrr)

                    # save
                    current_step += 1
                    if current_step % self.args.save_steps == 0:
                        # tf.logging.info("***** saving model to %s ****", self.args.output_dir)
                        ckpt_name_prefix = "last/last_models"
                        save_path = os.path.join(self.args.output_dir, ckpt_name_prefix)
                        self.model.saver.save(sess, save_path, global_step=100)
                        tf.logging.info("*****【%s】, training_step: %d 【%s】, train_loss: %f", datetime.now().strftime("%H:%M:%S"), current_step, str(100*current_step/self.num_train_steps)+"%", loss/(j+1))
 

                # 保存训练日志
                patten='%Y%m%d_%H_%M_%S'
                cur_time = time.strftime(patten, time.localtime(time.time()))
                with open("train_losses",  "a+") as f:
                    f.write(str(cur_time) + "\t" + str(i) + "\n" + str(train_losses) + "\n\n")
                with open("eval_losses",  "a+") as f:
                    f.write(str(cur_time) + "\t" + str(i) + "\n" + str(eval_losses) + "\n\n")
                with open("eval_mrrs",  "a+") as f:
                    f.write(str(cur_time) + "\t" + str(i) + "\n" + str(eval_mrrs) + "\n\n")
                
                # 每个epoch后保存模型
                try:
                    ckpt_name_prefix = "epoch/epoch{}_models".format(str(i))
                    save_path = os.path.join(self.args.output_dir, ckpt_name_prefix)
                    self.model.saver.save(sess, save_path, global_step=100)
                    tf.logging.info("*****【%s】, training_step: %d 【%s】, train_loss: %f", datetime.now().strftime("%H:%M:%S"), current_step, str(100*current_step/self.num_train_steps)+"%", loss/(j+1))
                except:
                    print("epoch 模型保存失败")
                    
                    
            tf.logging.info("total training time: %f", time.time()-starttime)

def set_seed(seed):
    '''
    随机种子设置
    :param seed:
    :return:
    '''
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

if __name__ == "__main__":
    ok_path = "./ok_path.txt"
    if os.path.exists(ok_path):
        os.remove(ok_path)
    import random
    set_seed(2022)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_path", type=str, default='/mnt/search/rank/demo/tianchi_ranker/bert_pretrain/bert/tmp/best_model/model.ckpt-46000', help="The path of bert_model") 
    parser.add_argument("--bert_config_path", type=str, default='/mnt/search/gen/simbert/model/chinese_L-12_H-768_A-12/bert_config.json', help="The path of bert_config file") 
    parser.add_argument("--bert_vocab_path", type=str, default='/mnt/search/gen/simbert/model/chinese_L-12_H-768_A-12/vocab.txt', help="The path of vocab.") 
    parser.add_argument("--output_dir", type=str, default="./v4", help="The path of checkpoint you want to save") 
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_proportion", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=50000)
    parser.add_argument("--query_ids_file_path", type=str, default="/mnt/search/recall2/WenTianSearch/data4rank/train.query.json", help="It's a json file, the result of converting text to ids, and the format is {'qid': xx, 'input_ids': xx}")
    parser.add_argument("--trainset_dir", type=str, default="/mnt/search/recall2/WenTianSearch/data4rank/0515_14_20_50", help="The directory of trainset, which contains queryid, golden_id, a series of docids.")
    parser.add_argument("--corpus_ids_file_path", type=str, default="/mnt/search/recall2/WenTianSearch/data4rank/corpus.json", help="Its format is the same as query_ids_file_path.") 
    parser.add_argument("--dev_ids_file_path", type=str, default="/mnt/search/rank/demo/tianchi_ranker/data/data_dev_ids", help="Its format is the same as query_ids_file_path.") 
    parser.add_argument("--is_training", type=bool, default=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    tf.logging.set_verbosity(tf.logging.INFO)
    trainer = Trainer(args)
    trainer.train()

    
    with open(ok_path, "w") as f:
        f.write(ok_path)
