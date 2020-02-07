
import pandas as pd 
import os 
import numpy as np 
import math 
import random
import tensorflow as tf

class DataProcessing:
    def __init__(self,file='data/ratings.csv'):

        self.uiscores=pd.read_csv(file)
        self.user_ids=set(self.uiscores["UserID"].values)
        self.item_ids = set(self.uiscores["MovieID"].values)

        # 用户对电影评分字典
        self.users_rating_dict = {}
        for user in self.user_ids:
            self.users_rating_dict.setdefault(str(user),{})
    
        with open(file,"r") as fr:
            for line in fr.readlines():
                if not line.startswith("UserID"):
                    (user,item,rate) = line.split(',')[:3]
                    self.users_rating_dict[user][item] = int(rate)

        self.all_length = len(self.uiscores)

    def get_train_data(self,batch_size,num_epochs):
        dataset = tf.data.Dataset.from_generator(self.get_train_pos_item,output_types=(tf.int32,tf.int32,tf.float32,tf.float32,tf.float32))
        dataset = dataset.map(self._parse_train,num_parallel_calls=8)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        user_id,item_id,rate = iterator.get_next()
        user_id.set_shape(batch_size)
        item_id.set_shape(batch_size)
        rate.set_shape(batch_size)
        return user_id,item_id,rate

    def get_test_data(self,batch_size=1000):
        dataset = tf.data.Dataset.from_generator(self.get_test_pos_item,output_types=(tf.int32,tf.int32,tf.float32,tf.float32,tf.float32))
        dataset = dataset.map(self._parse_test,num_parallel_calls=8)
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size,drop_remainder=True)
        iterator = dataset.make_one_shot_iterator()
        user_id,item_id,rate,min_value,max_value = iterator.get_next()
        user_id.set_shape(batch_size)
        item_id.set_shape(batch_size)
        rate.set_shape(batch_size)
        min_value.set_shape(batch_size)
        max_value.set_shape(batch_size)
        return user_id,item_id,rate,min_value,max_value

    def get_test_pos_item(self):
        users=random.sample(self.user_ids,10)
         
        for user in users:
            user_item_ids = set(self.uiscores[self.uiscores['UserID'] == user]['MovieID'])
            ratings = self.uiscores[self.uiscores['UserID'] == user]['Rating'].values
            min_values = min(ratings)
            max_values = max(ratings)
            
            for item_id in user_item_ids:
                
                r=self.uiscores[(self.uiscores['UserID'] == user)
                                & (self.uiscores['MovieID']==item_id)]["Rating"].values[0]
                yield (int(user),int(item_id),float(int(r)),float(int(min_values)),float(int(max_values)))
                
    def _parse_test(self,user_id,item_id,rate,min_value,max_value):
        return user_id,item_id,(rate-min_value)/max_value,min_value,max_value   
    def _parse_train(self,user_id,item_id,rate,min_value,max_value):
        return user_id,item_id,(rate-min_value) /max_value

    def get_train_pos_item(self):
        # 对用户有行为电影评分归一化
        

        for user_id in list(self.user_ids):
            ratings = self.uiscores[self.uiscores['UserID'] == user_id]['Rating'].values
            min_values = min(ratings)
            max_values = max(ratings)
            pos_item_ids = set(self.uiscores[self.uiscores['UserID'] == user_id]['MovieID'])
            for item in pos_item_ids:
                rating = self.users_rating_dict[str(user_id)][str(item)]
                yield (int(user_id),int(item),float(int(rating)),float(int(min_values)),float(int(max_values)))  

class LFM:
    def __init__(self,train_log='lfm_logs/graph',save_path='lfm_logs/checkpoint/model'):
        self.class_count = 5
        self.iter_count = 1
        self.batch_size = 64
        self.lr = 0.02
        self.lam = 0.01 
        process = DataProcessing()
        self.data_length = process.all_length
        self.user_length = max(process.uiscores["UserID"].values) + 1
        self.item_length = max(process.uiscores["MovieID"].values) + 1
        self.train_user_id,self.train_item_id,self.train_rate = process.get_train_data(batch_size=self.batch_size,num_epochs=self.iter_count)
        self.test_user_id,self.test_item_id,self.test_rate,self.test_min,self.test_max = process.get_test_data()


        self.train_log = os.path.join(os.path.dirname(os.path.abspath(__file__)),train_log)
        self.save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),save_path)

        save_parent_path = os.path.split(self.save_path)[0]
        if not os.path.exists(self.train_log):
            os.makedirs(self.train_log)
        
        if not os.path.exists(save_parent_path):
            os.makedirs(save_parent_path)
        
    def init_model(self):
        self.p = tf.get_variable('p',shape=[self.user_length,self.class_count],
                                initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))

        self.q = tf.get_variable('q',shape=[self.item_length,self.class_count],
                                initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))

    def build_model(self,user_id,item_id):
        r = tf.matmul(tf.gather(self.p,user_id),tf.transpose(tf.gather(self.q,item_id)))
        logits = tf.nn.sigmoid(tf.diag_part(r))
        return logits
    
    
    def build_loss(self,logits,y):

        loss = tf.reduce_mean(tf.square(y-logits)) + self.lam * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        tf.summary.scalar('loss',loss)
        self.global_step = tf.Variable(0,trainable=False)
        
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.minimize(loss,global_step=self.global_step)
        return loss,train_op

    def train(self):
        self.init_model()
        train_logits = self.build_model(self.train_user_id,self.train_item_id)
        train_loss, train_op = self.build_loss(train_logits,self.train_rate)
        sess = tf.Session()
        saver = tf.train.Saver(max_to_keep=5)
        sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.train_log,sess.graph)

        for e in range(self.iter_count):

            for step in range(math.floor(self.data_length/self.batch_size)):
                _,summary,global_steps,_ = sess.run([train_loss,merged,self.global_step,train_op])
                train_writer.add_summary(summary,global_step=global_steps)
                if step % 100 == 0:
                    test_logits = self.build_model(self.test_user_id,self.test_item_id)
                    test_loss = tf.reduce_mean(tf.square((self.test_rate - test_logits)*(self.test_max-self.test_min)+self.test_min))
                    loss = sess.run(test_loss)
                    print('Epoch: {} | step: {}  loss: {} '.format(e+1,step,loss))

                    saver.save(sess,self.save_path,global_step=global_steps)
                 
        sess.close()
   

if __name__ == "__main__":
    lfm = LFM()
    lfm.train()
   





    