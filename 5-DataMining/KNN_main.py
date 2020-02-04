import numpy as np 
from sklearn import neighbors

class KNN_test:
    def __init__(self,k):
        self.k = k
    
    def createData(self):
        features = np.array([[180,76],[158,43],[176,78],[161,49]])
        labels = ['男','女','男','女']
        return features,labels
    def Norm(self,data):
        maxs = np.max(data,axis=0)
        mins = np.min(data,axis=0)
        new_data = (data-mins)/(maxs-mins)
        return new_data,mins,maxs
    
    def __call__(self,one,data,labels):

        clf = neighbors.KNeighborsClassifier(self.k)
        new_data,mins,maxs = self.Norm(data)
        clf.fit(new_data,labels)
        new_one = (one-mins)/(maxs-mins)
        distance, nearests = clf.kneighbors(one,self.k)

        for near,dis in zip(nearests[0],distance[0]):
            print('最近点值： {} label: {} | 距离： {}'.format(data[near],labels[near],dis))  

        prediction = clf.predict(new_one)

        return prediction

if __name__ == "__main__":
    knntest = KNN_test(3)
    feature,labels = knntest.createData()

    one = np.array([[176,76]])

    result = knntest(one,feature,labels)
    print("数据 {} 的预测性别为 : {}".format(one, result[0]))   