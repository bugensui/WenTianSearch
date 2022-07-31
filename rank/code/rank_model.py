import tensorflow as tf
import os
import argparse
import numpy as np

from bert import modeling
from bert import optimization

# 第三方包
import utils

class Ranker(object):
    """This model is just for training."""
    def __init__(self, bert_config_path, is_training, num_train_steps=None, num_warmup_steps=None, learning_rate=1e-5):
        self.bert_config_path = bert_config_path
        self.learning_rate = learning_rate
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = num_warmup_steps
        self.is_training = is_training

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        self.token_type_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="token_type_ids")
        self.label_ids = tf.placeholder(dtype=tf.int32, shape=[None,], name="label_ids")
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=None, name="batch_size")
        
        self.create_model()
        self.init_saver()

    def create_model(self):
        num_labels = 2
        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_path)
        model = modeling.BertModel(config=self.bert_config, is_training=self.is_training, 
                                   input_ids=self.input_ids, input_mask=self.input_mask, token_type_ids=self.token_type_ids,
                                   use_one_hot_embeddings=False)
        output_layer = model.get_pooled_output()
        self.output_layer = output_layer
        hidden_size = output_layer.shape[-1].value

        # 获取输出层的weights和bias
        with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
            output_weights = tf.get_variable("output_weights", [num_labels-1, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02)) # num_label=2
            output_bias = tf.get_variable("output_bias", [num_labels-1], initializer=tf.zeros_initializer())

        # 相关性打分
        with tf.variable_scope("loss"):
            if self.is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            self.logits = tf.nn.bias_add(logits, output_bias)
            self.probabilities = tf.nn.sigmoid(self.logits)
            self.score = tf.identity(self.probabilities, name="score")

        # 计算loss（反向传播）
        if self.is_training:
            with tf.name_scope("train_op"):
                self.label_ids = tf.cast(self.label_ids, dtype=tf.float32)
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.label_ids, (-1, self.logits.shape[-1].value))))
                self.train_op = optimization.create_optimizer(
                        self.loss, self.learning_rate, self.num_train_steps, self.num_warmup_steps, use_tpu=False)


    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch):
        batch_size = len(batch["input_ids"])
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_mask: batch["input_mask"],
                     self.token_type_ids: batch["token_type_ids"],
                     self.label_ids: batch["label_ids"],
                     self.batch_size: batch_size}
        
        """
        # 在这加入对抗训练
        if fgm == False:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                loss, logits, prob = sess.run([self.loss, self.logits, self.probabilities], feed_dict=feed_dict)
                grads_and_vars = utils.compute_gradients(loss, self.train_op)  # 计算梯度（没回传）
            
            # loss对embedding的梯度 获取word_embeddings对应的梯度
            embedding_gradients, embeddings = utils.find_grad_and_var(grads_and_vars, layer_name)

            r = tf.multiply(epsilon, embedding_gradients / (tf.norm(embedding_gradients) + 1e-9)) # 计算扰动
            attack_op = embeddings.assign(embeddings + r)  # 攻击word_embeddings

            # restore
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE), tf.control_dependencies([attack_op]):
                loss, logits, prob = sess.run([self.loss, self.logits, self.probabilities], feed_dict=feed_dict)  
                attack_grad_and_vars = utils.compute_gradients(adv_outputs['loss'], optimizer)  # 计算被攻击后的梯度（没回传）
                restore_op = embeddings.assign(embeddings - r)  # 恢复 word_embeddings

            # sum up
            with tf.control_dependencies([restore_op]):
                grads_and_vars = utils.average_grads_and_vars([grads_and_vars, attack_grad_and_vars])  # 累加原始梯度和被攻击后的梯度
        
            # backward
        """
 
        
        # print("train", sess.run([self.train_op, self.loss, self.logits, self.probabilities], feed_dict=feed_dict))
        _, loss, logits, prob = sess.run([self.train_op, self.loss, self.logits, self.probabilities], feed_dict=feed_dict)
        return loss, logits, prob

    def eval(self, sess, batch):
        batch_size = len(batch["input_ids"])
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_mask: batch["input_mask"],
                     self.token_type_ids: batch["token_type_ids"],
                     self.label_ids: batch["label_ids"],
                     self.batch_size: batch_size}
        # print("eval", sess.run([self.loss, self.logits, self.probabilities], feed_dict=feed_dict))
        loss, logits, prob = sess.run([self.loss, self.logits, self.probabilities], feed_dict=feed_dict)
        # print(len(prob))
        # 计算MRR
        mrr = 0
        pos_prob = prob[0]
        neg_probs = prob[1:]
        neg_probs = sorted(neg_probs)[::-1]
        for ind, neg_prob in enumerate(neg_probs):
            if ind >= 10: break
            if pos_prob > neg_prob:
                mrr = 1 / (ind + 1)
                break

        return loss, logits, prob, mrr

    

