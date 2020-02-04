import numpy as np
import math 
from sklearn import tree
import graphviz
from sklearn.externals.six import StringIO
import pydotplus
class DecisionTree:
    def __init__(self):
        pass 

    def __call__(self,data,features,x):
        '''
        默认使用sklearn都决策树
        '''
        clf = tree.DecisionTreeClassifier(criterion='entropy')
        '''
        各参数说明：
        criterion:1）输入”entropy“，使用信息熵（Entropy）

                  2）输入”gini“，使用基尼系数（Gini Impurity）
        random_state：用来设置分枝中的随机模式的参数
        splitter：也是用来控制决策树中的随机选项的，
                  输入”best"，决策树在分枝时虽然随机，但是还是会优先选择更重要的特征进行分枝（重要性可以通过属性feature_importances_查看）
                  输入“random"，决策树在分枝时会更加随机，树会因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合。这也是防止过拟合的一种方式。
        max_depth：限制树的最大深度，超过设定深度的树枝全部剪掉
        min_samples_leaf：限定，一个节点在分枝后的每个子节点都必须包含至少min_samples_leaf个训练样本，否则分枝就不会发生
        min_samples_split：限定，一个节点必须要包含至少min_samples_split个训练样本，这个节点才允许被分枝，否则分枝就不会发生
        max_features：用作树的”精修“，限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃。
        min_impurity_decrease：限制信息增益的大小，信息增益小于设定数值的分枝不会发生
        class_weight：参数对样本标签进行一定的均衡，给少量的标签更多的权重，让模型更偏向少数类，向捕获少数类的方向建模
        min_ weight_fraction_leaf：这个基于权重的剪枝参数来使用
        '''
        clf.fit(np.array(data)[:,:-1],np.array(data)[:,-1])

        # 可视化树
        dot_data=StringIO()

        tree.export_graphviz(clf,out_file=dot_data
                                , feature_names=features
                                , class_names=["yes", "no"]
                                , filled=True
                                , rounded=True)
 
        graph = graphviz.Source(dot_data.getvalue())
        graph.view()

        prediction = clf.predict(np.array([x]))
 
        return prediction

    #加载数据
    def loadData(self):
        # 天气晴(2),阴(1),雨(0);温度炎热(2),适中(1),寒冷(0);湿度高(1),正常(0)
        # 风速强(1),弱(0);进行活动(yes),不进行活动(no)
        # 创建数据集
        data = [
            [2, 2, 1, 0, "yes"],
            [2, 2, 1, 1, "no"],
            [1, 2, 1, 0, "yes"],
            [0, 0, 0, 0, "yes"],
            [0, 0, 0, 1, "no"],
            [1, 0, 0, 1, "yes"],
            [2, 1, 1, 0, "no"],
            [2, 0, 0, 0, "yes"],
            [0, 1, 0, 0, "yes"],
            [2, 1, 0, 1, "yes"],
            [1, 2, 0, 0, "no"],
            [0, 1, 1, 1, "no"],
        ]
        # 分类属性
        features = ["天气", "温度", "湿度", "风速"]
        return data, features

    def ShannonEnt(self,data):
        #计算信息熵
        numData = len(data)
        labelCounts = {}
        for feature in data:
            oneLabel = feature[-1] #获取标签
            labelCounts[oneLabel] = labelCounts.get(oneLabel,0)+1

        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numData

            shannonEnt -= prob * math.log2(prob)
        return shannonEnt
    
    def splitData(self,data,axis,value):
        '''
        划分数据集
        Args:
            data:待划分的数据集
            axis:划分数据集特征的维度下标
            value:划分维度所取的值
        Return:
            retData:除去已分特征的数据
        '''
        retData = []
        for feature in data:
            if feature[axis] == value:
                reducedFeature = feature[:axis]
                reducedFeature.extend(feature[axis+1:])
                retData.append(reducedFeature)
        return retData

    def chooseBestFeatureToSplit(self,data):
        '''
        选择使信息增益最大的特征进行分割，返回特征维度下标
        '''
        #所有待分割的特征
        numFeature = len(data[0]) - 1

        # data数据信息熵，用来计算信息增益
        baseEntropy = self.ShannonEnt(data)

        # 用来标定最大信息增益和相应特征下标
        bestInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeature):
            # i 特征所有数据的取值
            featureList = [result[i] for result in data]
            # 除去重复值
            uniqueFeatureList = set(featureList)

            # 用来计算条件信息熵
            newEntropy = 0.0
            for value in uniqueFeatureList:
                # 返回第i各特征值为value的数据，其中数据特征除去i
                splitDataSet = self.splitData(data,i,value)

                # 计算条件信息熵
                prob = len(splitDataSet) / float(len(data))
                newEntropy += prob * self.ShannonEnt(splitDataSet)
            
            # 信息增益
            infoGain = baseEntropy - newEntropy
            # 挑选最大的
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i 
        return bestFeature

    def majorityCnt(self,labelsList):
        '''
        若最后没有特征可分，且所有label不统一，选择次数最多的
        Args:
            labelsList:剩余待分数据label
        Return：
            出现次数最多的label值
        '''
        labelsList = np.array(labelsList)
        return np.argmax(np.bincount(labelsList))

    def createTree(self,data,features):
        '''
        创建决策树
        '''
        features = list(features)
        labelsList = [line[-1] for line in data]

        # 划分后的类别都相同停止划分返回类别
        if labelsList.count(labelsList[0]) == len(labelsList):
            return labelsList[0]
        # 若最后没有特征可分，且所有label不统一，选择次数最多的返回
        if len(data[0]) == 1:
            return self.majorityCnt(labelsList)

        # 得到划分数据集都特征
        bestFeature = self.chooseBestFeatureToSplit(data)
        bestFeatLabel = features[bestFeature]
        myTree = {bestFeatLabel:{}}

         # 清空features[bestFeat],在下一次使用时清零
        del (features[bestFeature])
        featureValues = [example[bestFeature] for example in data]
        uniqueFeatureValues = set(featureValues)
        for value in uniqueFeatureValues:
            subFeatures = features[:]
            # 每个特征所取都值继续划分决策树
            myTree[bestFeatLabel][value] = self.createTree(
                self.splitData(data,bestFeature,value),subFeatures
            )
        return myTree

    def predict(self,tree,features,x):
        '''
        预测x 属于哪一个类别

        '''
        # 由根节点不断向下匹配，直到匹配叶子结点返回类别
        for key1 in tree.keys():
            secondDict = tree[key1]
            featIndex = features.index(key1)

            for key2 in secondDict.keys():
                if x[featIndex] == key2:
                    if type(secondDict[key2]).__name__ == 'dict':
                        classLabel = self.predict(secondDict[key2],features,x)
                    else:
                        classLabel = secondDict[key2]
        return classLabel


if __name__ == "__main__":
    dtree = DecisionTree()
    data,features = dtree.loadData()
    # # python手编
    # myTree = dtree.createTree(data,features)
    # print('决策树结构：')
    # print(myTree)
    # label = dtree.predict(myTree,features,[1,1,1,0])
    # print("新数据[1,1,1,0]对应的是否要进行活动为:{}".format(label))
    # sklearn 决策树为二叉树
    prediction = dtree(data,features,[1,1,1,0])
    print("新数据[1,1,1,0]对应的是否要进行活动为:{}".format(prediction[0]))
    
