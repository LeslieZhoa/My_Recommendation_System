import numpy as np 

class NaiveBayesian:
    def __init__(self,alpha):
        self.classP = dict()
        self.classP_feature = dict()
        self.alpha = alpha # 平滑值

    def createData(self):
        data = np.array(
            [
                [320, 204, 198, 265],
                [253, 53, 15, 2243],
                [53, 32, 5, 325],
                [63, 50, 42, 98],
                [1302, 523, 202, 5430],
                [32, 22, 5, 143],
                [105, 85, 70, 322],
                [872, 730, 840, 2762],
                [16, 15, 13, 52],
                [92, 70, 21, 693],
            ]
        )
        labels = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
        return data, labels

    def guassian(self,mu,sigma,x):
        #计算高斯分布函数值
        return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    
    def calMuAndSigma(self,feature):
        # 计算某个特征列对应均值和标准差
        mu = np.mean(feature)
        sigma = np.std(feature)
        return (mu, sigma)

    def train(self,data,labels):
        # 训练朴素贝叶斯算法模型
        numData = len(labels)
        numFeatures = len(data[0])
        # 是异常用户概率
        self.classP[1] = (
            (sum(labels) + self.alpha) * 1.0 / (numData + self.alpha * len(set(labels)))
        )

        # 不是异常用户概率
        self.classP[0] = 1 - self.classP[1]
        # 用来存放每个label下每个特征标签下对应的高斯分布中的均值和方差
        # { label1:{ feature1:{ mean:0.2, var:0.8 }, feature2:{} }, label2:{...} }
        self.classP_feature = dict()
        # 遍历每个特征标签
        for c in set(labels):
            self.classP_feature[c] = {}
            for i in range(numFeatures):
                feature = data[np.equal(labels,c)][:,i]
                self.classP_feature[c][i] = self.calMuAndSigma(feature)

    def predict(self,x):
        # 预测新用户是否是异常用户
        label = -1
        maxP = 0
        # 遍历所有label值
        for key in self.classP.keys():
            label_p = self.classP[key]
            currentP = 1.0
            feature_p = self.classP_feature[key]
            j = 0
            for fp in feature_p.keys():
                currentP *= self.guassian(feature_p[fp][0],feature_p[fp][1],x[j])
                j += 1
            
            if currentP * label_p > maxP:
                maxP = currentP * label_p
                label = key 
        return label 

if __name__ == "__main__":
    nb = NaiveBayesian(1.0)
    data, labels = nb.createData()
    nb.train(data,labels)
    label = nb.predict(np.array([134,84,235,349]))
    print("未知类型用户对应的行为数据为：[134,84,235,349]，该用户的可能类型为：{}".format(label))
