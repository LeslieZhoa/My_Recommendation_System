#参考https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/word2vec
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile

import numpy as np
import tensorflow as tf
import argparse
import sys

def read_data(filename):
    '''
    读取filename中的单词
    filename可以存储的是一篇文章
    data返回的是单个单词
    '''
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_datasets(words,vocabulary_size=50000):
    '''
    返回词字典等数据
    Args：
        words：来自read_data 的返回值
    Returns:
        dictionary: 词字典，key是单词，value是从0开始的index，频率最大的词为0，依次类推
        data: words中每个单词对应的index
        count:[word,出现频数]
        reverse_dictionary: key是index，value是单词
    '''
    count = [['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()
    for word,_ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary

def generate_batch(batch_size,num_skips,skip_window,data):
    '''
    生成训练数据 假设相邻词为['a','b','c'] 生成的input和label对可能是['b','a'],['b','c']
    Args:
        num_skips:重复采样
        skip_window:上下文中采样对范围
    Return:
        batch,labels: 单词对应对index
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2* skip_window
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    for i in range(batch_size//num_skips):
        target = skip_window 
        target_to_avoid = [skip_window]
        for j in range(num_skips): #重复采样
            while target in target_to_avoid: 
                #找到除中心词以及找过的词之外的词
                target = random.randint(0,span-1)
            target_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]#一直是中心词
            labels[i * num_skips + j,0] = buffer[target]
        buffer.append(data[data_index])#更新数据
        data_index = (data_index + 1) % len(data)
    return batch, labels

def train(opt):
    #读取数据
    words = read_data(opt.filename)
    #构建词字典
    data,count,dictionary,reverse_dictionary = build_datasets(words,opt.vocabulary_size)

    graph = tf.Graph()

    with graph.as_default():
        # input data
        train_inputs = tf.placeholder(tf.int32,shape=[opt.batch_size])
        train_labels = tf.placeholder(tf.int32,shape=[opt.batch_size,1])
        
        # vaild data
        valid_example = np.random.choice(opt.valid_window,opt.valid_size,replace=False)
        valid_dataset = tf.constant(valid_example,dtype=tf.int32)

        embeddings = tf.Variable(
            tf.random_uniform([opt.vocabulary_size,opt.embedding_size],-1.0,1.0),name='embedding')
        embed = tf.nn.embedding_lookup(embeddings,train_inputs)

        nce_weights = tf.Variable(
            tf.truncated_normal([opt.vocabulary_size,opt.embedding_size],
            stddev=1.0/math.sqrt(opt.embedding_size)))
        nce_biases = tf.Variable(tf.zeros(opt.vocabulary_size))
        '''
        计算nce loss
        步骤：
        1.随机选取与label的index不相同的index-->neg，其中高频词的几率更大
        2.将label与neg的index concat在一起-->all_index，选取nce_weights与nce_biases的all_index部分
            与embed做矩阵运算得到logits.对于每个batch all_index的第一维为true,其他维为false，所以对应的label
            应为[1,0,0,...]
        3.将label和logits做交叉熵
        '''
        loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            inputs=embed,
            labels=train_labels,
            num_sampled=opt.num_sampled,
            num_classes=opt.vocabulary_size))
        optimizer = tf.train.GradientDescentOptimizer(opt.lr).minimize(loss)
        tf.summary.scalar('loss', loss)
        #计算cosine距离，找出与val距离最近的单词
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
        normalized_embeddings = embeddings / norm 
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
        similarity = tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)
        
        merged_summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as sess :
        saver = tf.train.Saver(tf.all_variables(),max_to_keep=100)
        train_writer=tf.summary.FileWriter(opt.graph_path,sess.graph)
        init.run()
        print('Initialized')
        average_loss = 0
        for step in range(opt.num_steps):
            batch_inputs,batch_labels = generate_batch(opt.batch_size,
            opt.num_skips,opt.skip_window,data)
            feed_dict = {train_inputs:batch_inputs,train_labels:batch_labels}
            _,loss_val,train_summary = sess.run([optimizer,loss,merged_summary],feed_dict)
            average_loss += loss_val
            train_writer.add_summary(train_summary, step)
            if step % 2000 == 0 :
                if step > 0:
                    average_loss /= 2000
                print('Average loss at step ',step,': ',average_loss)
                average_loss = 0
                saver.save(sess, os.path.join(opt.model_path,'word2vec'),global_step=step)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(opt.valid_size):
                    valid_word = reverse_dictionary[valid_example[i]]
                    top_k = opt.top_k
                    nearest = (-sim[i,:]).argsort()[1:top_k+1]
                    log_str = 'Nearest to %s:'%valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s,"%(log_str,close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()
    return final_embeddings,reverse_dictionary


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

    plt.savefig(filename)


def arg_parser(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, help='the data path',
                        default= './text8.zip')

    parser.add_argument('--batch_size', type=int, help='',
                        default=64)

    parser.add_argument('--num_skips', type=int, help='',
                        default=2)

    parser.add_argument('--skip_window', type=int, help='',
                        default=1)

    parser.add_argument('--valid_window', type=int, help='Only pick dev samples in the head of the distribution.', 
                        default=100)

    parser.add_argument('--valid_size', type=int, help='Random set of words to evaluate similarity on.',
                        default=16)

    parser.add_argument('--vocabulary_size', type=int, help='',
                        default=50000)

    parser.add_argument('--embedding_size', type=int, help='', 
                        default=128)

    parser.add_argument('--num_sampled', type=int, help='负样本采样数', 
                        default=40)
    
    parser.add_argument('--lr', type=int, help='', 
                        default=1)

    parser.add_argument('--num_steps', type=int, help='总训练长度', 
                        default=100001)
    
    parser.add_argument('--model_path', type=str, help='', 
                        default='logs/model')
    
    parser.add_argument('--graph_path', type=str, help='', 
                        default='logs/graph')
    
    parser.add_argument('--top_k', type=int, help='', 
                        default=4)
    


    return parser.parse_args(args)

if __name__ == "__main__":
    
    data_index = 0
    opt = arg_parser(sys.argv[1:])
    if not os.path.exists(opt.model_path):
        os.makedirs(opt.model_path)
    
    if not os.path.exists(opt.graph_path):
        os.makedirs(opt.graph_path)
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        final_embeddings,reverse_dictionary = train(opt)
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels)

    except ImportError:
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")

