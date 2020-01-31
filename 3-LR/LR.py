from utils import Process
import tensorflow as tf
import os
import math
import sys
import argparse

class LR:
    def __init__(self,opt):
        self.file = os.path.join(os.path.dirname(os.path.abspath(__file__)),opt.file)
        process = Process(self.file,opt)
        self.inputs,self.labels,_ = process.get_data()
        # feature length
        self.p = process.feature_length
        self.lr = opt.lr # 学习率
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
            self.y_out = tf.add(tf.sparse_tensor_dense_matmul(self.inputs,w1),b)


    def build_loss(self):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.one_hot(self.labels,depth=2),logits=self.y_out)
        train_var_list=[v for v in tf.trainable_variables()]
        self.loss = tf.reduce_mean(cross_entropy) + self.reg_l2 * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])
        tf.summary.scalar('loss',self.loss)
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_out,1),tf.int32),self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',self.accuracy)


        self.global_step = tf.Variable(0,trainable=False)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
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
            for step in range(math.ceil(self.data_length/self.opt.batch_size)):
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

   

    parser.add_argument('--lr', type=float, help='学习率', 
                        default=0.01)


    parser.add_argument('--reg_l2', type=float, help='',
                        default=1e-4)

    parser.add_argument('--batch_size', type=int, help='', 
                        default=64)

    parser.add_argument('--num_epochs', type=int, help='', 
                        default=5)
    
    parser.add_argument('--print_every', type=int, help='', 
                        default=100)

    parser.add_argument('--train_log', type=str, help='', 
                        default='logs/graph')

    parser.add_argument('--save_path', type=str, help='', 
                        default='logs/checkpoint/model')

    parser.add_argument('--fields_dict_path', type=str, help='', 
                        default='avazu-ctr-prediction/fields_dict.npy')

    return parser.parse_args(args)

if  __name__ == "__main__":
    opt = arg_parser(sys.argv[1:])
    train_lr = LR(opt)
    # train the model
    train_lr.train_model()