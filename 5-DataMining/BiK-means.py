import numpy as np 
import pandas as pd 
import random 
from sklearn.cluster import KMeans

class K_means:
    def __init__(self):
        pass 

    def loadData(self,file):
        #加载数据集
        return pd.read_csv(file,header=0,sep=',')

    def __call__(self,data,k,maxIters):
        km = KMeans(n_clusters=k,init='random',max_iter=maxIters)
       
        km.fit(np.reshape(data,[-1,1]))
        return km.cluster_centers_,km

    def filterAnomalyValue(self,data):
        #去除异常值，使用正态分布，保证最大异常值为5000，最小为1
        upper = np.mean(data['price']) + 3 * np.std(data['price'])
        lower = np.mean(data['price']) - 3 * np.std(data['price'])
        upper_limit = upper if upper < 5000 else 5000
        lower_limit = lower if lower > 1 else 1
        print('最大异常值：{},最小异常值：{}'.format(upper_limit,lower_limit))

        # 过滤掉大于最大异常值和小于最小异常值掉
        newData = data[(data['price']<upper_limit) &
                        (data['price']>lower_limit)]
        
        return newData,upper_limit,lower_limit

    def initCenters(self,values,K,Cluster):
        #初始化簇类中心
        random.seed(100)
        oldCenters = list()
        for i in range(K):
            index = random.randint(0,len(values))
            Cluster.setdefault(i,{})
            Cluster[i]['center'] = values[index]
            Cluster[i]['values'] = []

            oldCenters.append(values[index])
        return oldCenters,Cluster

    def distance(self,price1,price2):
        # 计算距离
        return np.emath.sqrt(pow(price1-price2,2))

    def k_means(self,data,K,maxIters):
        Cluster = dict() # 最终聚类结果
        oldCenters,Cluster = self.initCenters(data,K,Cluster)
        print('初始簇类中心：{}'.format(oldCenters))
        clusterChange = True
        i = 0
        while clusterChange:
            for price in data:
                minDistance = np.inf
                minIndex = -1
                for key in Cluster.keys():
                    dis = self.distance(price,Cluster[key]['center'])
                    if dis < minDistance:
                        minDistance = dis 
                        minIndex = key 
                Cluster[minIndex]['values'].append(price)

            newCenters = list()
            for key in Cluster.keys():
                newCenter = np.mean(Cluster[key]['values']) 
                Cluster[key]['center'] = newCenter
                newCenters.append(newCenter)

            print('第{}次迭代后掉簇族中心：{}'.format(i,newCenters))

            if oldCenters == newCenters or i > maxIters:
                clusterChange = False
            
            else:
                oldCenters = newCenters 
                i += 1
                # 删除记录值
                for key in Cluster.keys():Cluster[key]['values']=[]
        return Cluster

    def SSE(self,data,mean):
        # 计算对应sse值
        newData = np.mat(data) - mean 
        return (newData * newData.T).tolist()[0][0]

    def diKMeans(self,data,K=7):
        # 二分kMeans
        clusterSSEResult = dict()
        clusterSSEResult.setdefault(0,{})
        clusterSSEResult[0]['values'] = data 
        clusterSSEResult[0]['sse'] = np.inf 
        clusterSSEResult[0]['centers'] = np.mean(data)

        while len(clusterSSEResult) < K:
            maxSSE = -np.inf 
            # 找sse最大都簇进行kmeans
            for key in clusterSSEResult.keys():
                if clusterSSEResult[key]['sse'] > maxSSE:
                    maxSSE = clusterSSEResult[key]['sse']
                    maxSSEKey = key 
            clusterResult = self.k_means(
                clusterSSEResult[maxSSEKey]['values'],K=2,maxIters=200
            )

            #删除对应maxKey对应都值
            del clusterSSEResult[maxSSEKey]
            # 聚类后赋值
            clusterSSEResult.setdefault(maxSSEKey,{})
            clusterSSEResult[maxSSEKey]['center'] = clusterResult[0]['center']
            clusterSSEResult[maxSSEKey]['values'] = clusterResult[0]['values']
            clusterSSEResult[maxSSEKey]['sse'] = self.SSE(
                clusterResult[0]['values'],clusterResult[0]['center']
            )
            maxKey = max(clusterSSEResult.keys()) + 1
            clusterSSEResult.setdefault(maxKey,{})
            clusterSSEResult[maxKey]["center"]=clusterResult[1]["center"]
            clusterSSEResult[maxKey]["values"]=clusterResult[1]["values"]
            clusterSSEResult[maxKey]["sse"]=\
                self.SSE(clusterResult[1]["values"],clusterResult[1]["center"])

        return clusterSSEResult




if __name__ == "__main__":
    file = 'data/skuid_price.csv'
    km = K_means()
    data = km.loadData(file)
    newData,upper_limit,lower_limit = km.filterAnomalyValue(data)
    Cluster = km.diKMeans(newData['price'].values,K=7)
    centers = []
    for key in Cluster.keys():
        centers.append(Cluster[key]['center'])
    print('final center: ',centers)
