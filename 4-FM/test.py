from utils import Process
import tensorflow as tf
class Opt:
    def __init__(self):
        self.batch_size = 2
        self.num_epochs = 1
path = './avazu-ctr-prediction/train.csv'
process = Process(path,Opt())
a,b = process.get_data()

sess = tf.Session()
c,d = sess.run([a,b])
print('a： ',c)
print('b:  ',d)
