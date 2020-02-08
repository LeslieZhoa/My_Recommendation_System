'''
base on https://www.omegaxyz.com/2019/03/21/dbscan-python/
'''

from sklearn import datasets
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy
from sklearn.cluster import DBSCAN

class DBS_CAN:
    def __init__(self):
        self.X = np.array(self.loadDataSet())
        self.eps = 0.2
        self.min_Pts = 3 
        self.color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']


    def __call__(self,data):
        
 

        db = DBSCAN(eps=self.eps, min_samples=self.min_Pts).fit(data)
 
        labels = db.labels_ 
        cluster = []
        for label in labels:
            cluster.append(self.color[label])
        return cluster
 
    def loadDataSet(self,filename='data/dataSet.txt'):
        X1, y1 = datasets.make_circles(n_samples=100, factor=.6, noise=.02)
        X2, y2 = datasets.make_blobs(n_samples=50, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]], random_state=9)
        return np.concatenate((X1, X2))
    def find_neighbor(self,x1, x, eps):
        N = list()
        for i in range(x.shape[0]):
            temp = np.sqrt(np.sum(np.square(x1-x[i])))  # 计算欧式距离
            if temp <= eps:
                N.append(i)
        return set(N)
 
 
    def DBSCAN(self):
        X = self.X 
        eps = self.eps 
        min_Pts = self.min_Pts
        k = 0 # 数据属于哪个类别
        neighbor_list = []  # 用来保存每个数据的邻域
        omega_list = []  # 核心对象集合
        unvisited = set([x for x in range(len(X))])  # 未被访问数据点
        cluster = [self.color[0] for _ in range(len(X))]  # 聚类 默认初始属于类别0
        for i in range(len(X)):
            neighbor_list.append(self.find_neighbor(X[i], X, eps))
            if len(neighbor_list[-1]) >= min_Pts:
                omega_list.append(i)  # 将样本加入核心对象集合
        omega_list = set(omega_list)  # 转化为集合便于操作
        #画图
        # plt.figure()
        #打开交互模式
        plt.ion()
        while len(omega_list) > 0:
            unvisited_old = copy.deepcopy(unvisited)
            j = random.choice(list(omega_list))  # 随机选取一个核心对象
            k = k + 1 # 数据类别

            # 存放j的密度可达的点，且每次搜索Q内其他点的直接密度可达点就将该点remove掉，
            # Q为空表示j的密度可达点找完
            Q = list()
            Q.append(j)
            unvisited.remove(j)
            while len(Q) > 0:
                q = Q[0]
                Q.remove(q)
                if len(neighbor_list[q]) >= min_Pts:
                    delta = neighbor_list[q] & unvisited
                    deltalist = list(delta)
                    for i in range(len(delta)):
                        Q.append(deltalist[i])
                        unvisited = unvisited - delta

            # 标记参观过点的类别          
            Ck = unvisited_old - unvisited
            Cklist = list(Ck)
            for i in range(len(Ck)):
                cluster[Cklist[i]] = self.color[k]
                # 画图
                # 清除原有图像
                plt.cla()
                plt.scatter(X[:, 0], X[:, 1], c=cluster)
                # 暂停
                plt.pause(0.2)
            omega_list = omega_list - Ck
        # 关闭交互模式
        plt.ioff()

        # 图形显示
        plt.show()
            
        
            
        return cluster



begin = time.time()
dbs_can = DBS_CAN()
cluster = dbs_can.DBSCAN()
# cluster = dbs_can(dbs_can.X)
end = time.time()
print('time is : {}'.format(end-begin))
plt.figure()
plt.scatter(dbs_can.X[:, 0], dbs_can.X[:, 1], c=cluster)
plt.show()
