import numpy as np 
from sklearn import datasets
from sklearn.decomposition import PCA 

class PCATest:
    def __init__(self):
        pass
    def loadIris(self):
        data = datasets.load_iris()["data"]
        return data 
    def __call__(self,data,k=None,scale=None):
        if k != 0:
            pca = PCA(n_components=k)
        elif scale != 0:
            pca = PCA(scale)
        
        new_data = pca.fit_transform(data)

        return new_data,sum(pca.explained_variance_ratio_),pca

if __name__ == "__main__":
    pcatest = PCATest()
    data = pcatest.loadIris()
    
    k = 2
    scale = 0.98
    new_data,scale,pca = pcatest(data,k,scale)

    print('所保留的n个成分各自的方差百分比: ',scale)
    print("最终降维结果为:\n{}".format(new_data))
    # 得到重构数据
    print("最终重构结果为:\n{}".format( pca.inverse_transform(new_data)) )
