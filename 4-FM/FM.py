from utils import Process
import tensorflow as tf
import os
import math
import sys
import argparse

class FM:
    def __init__(self,opt):
        self.file = os.path.join(os.path.dirname(os.path.abspath(__file__)),opt.file)
        process = Process(self.file,opt)
        self.inputs,self.labels,_= process.get_data()
        # feature length
        self.p = process.feature_length
        self.k = opt.k # 参数v 的维度
        self.lr = opt.lr # 学习率
        self.reg_l1 = opt.reg_l1
        self.reg_l2 = opt.reg_l2

        self.data_length = process.data_length # 数据总量
        self.opt = opt

        self.train_log = os.path.join(os.path.dirname(os.path.abspath(__file__)),opt.train_log)
        self.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),opt.save_path)

        save_parent_path = os.path.split(self.save_path)[0]
        if not os.path.exists(self.train_log):
            os.makedirs(self.train_log)
        
        if not os.path.exists(save_parent_path):
            os.makedirs(save_parent_path)
       
    def build_model(self):
        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias',shape=[2],
                                initializer=tf.zeros_initializer())

            w1 = tf.get_variable('w1',shape=[self.p,2],
                                initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))

            # shape [None,2]
            self.linear_terms = tf.add(tf.sparse_tensor_dense_matmul(self.inputs,w1),b)
        
        with tf.variable_scope('interaction_layer'):
            v = tf.get_variable('v',shape=[self.p,self.k],
                                initializer=tf.truncated_normal_initializer(mean=0,stddev=0.01))
            
            # shape [None,1]
            # print('vvvvvvvvvvvvvv',self.labels.get_shape().as_list())
            tf_eye = tf.eye(num_rows=self.labels.get_shape().as_list()[0],num_columns=self.p)
            input_tensor = tf.matmul(tf.sparse_tensor_dense_matmul(self.inputs,tf.transpose(tf_eye)),tf_eye)
            
            self.interaction_terms = tf.multiply(0.5,tf.reduce_mean(
                                                tf.subtract(
                                                    tf.pow(tf.sparse_tensor_dense_matmul(self.inputs,v),2),
                                                    tf.matmul(tf.pow(input_tensor,2),tf.pow(v,2))),
                                                    1,keep_dims=True))
            
            # shape [None,2]
            self.y_out = tf.add(self.linear_terms,self.interaction_terms)
            self.y_out_prob = tf.nn.softmax(self.y_out)


    def build_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,logits=self.y_out)
        self.loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss',self.loss)
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_out,1),tf.int32),self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',self.accuracy)


        self.global_step = tf.Variable(0,trainable=False)
        '''
        更新方法主要用于广告点击预测,广告点击预测通常千万级别的维度,因此有巨量的稀疏权重.其主要特点是将接近0 的权重直接置0,这样
        计算时可以直接跳过,从而简化计算.这个方法已经验证过在股票数据上较有效。
        '''
        optimizer = tf.train.FtrlOptimizer(self.lr,l1_regularization_strength=self.reg_l1,
                                            l2_regularization_strength=self.reg_l2)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss,global_step=self.global_step)
    

    def train_model(self):

        # build net
        self.build_model()
        # build loss
        self.build_loss()
        
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=5)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.train_log,sess.graph)

        for e in range(self.opt.num_epochs):
            for step in range(math.floor(self.data_length/self.opt.batch_size)):
                loss, accuracy, summary,global_steps, _ = sess.run([
                    self.loss, self.accuracy, merged, self.global_step, self.train_op
                ])

                train_writer.add_summary(summary,global_step=global_steps)

                if global_steps % self.opt.print_every == 0:
                    print('Epoch: {} | step: {}  loss: {} | accuracy: {}'.format(e+1,step+1,loss,accuracy))

                    saver.save(sess,self.save_path,global_step=global_steps)
        coord.request_stop()
        coord.join(threads)            
        sess.close()

def arg_parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='数据存放目录',
                        default= 'avazu-ctr-prediction/train.csv')

    parser.add_argument('--k', type=int, help='参数维度',
                        default=40)

    parser.add_argument('--lr', type=float, help='学习率', 
                        default=0.01)

    parser.add_argument('--reg_l1', type=float, help='',
                        default=2e-2)

    parser.add_argument('--reg_l2', type=float, help='',
                        default=0)

    parser.add_argument('--batch_size', type=int, help='', 
                        default=64)

    parser.add_argument('--num_epochs', type=int, help='', 
                        default=5)
    
    parser.add_argument('--print_every', type=int, help='', 
                        default=100)

    parser.add_argument('--train_log', type=str, help='', 
                        default='FM_logs/graph')

    parser.add_argument('--save_path', type=str, help='', 
                        default='FM_logs/checkpoint/model')

    parser.add_argument('--fields_dict_path', type=str, help='', 
                        default='avazu-ctr-prediction/fields_dict.npy')

    return parser.parse_args(args)

if  __name__ == "__main__":
    opt = arg_parser(sys.argv[1:])
    train_fm = FM(opt)
    # train the model
    train_fm.train_model()