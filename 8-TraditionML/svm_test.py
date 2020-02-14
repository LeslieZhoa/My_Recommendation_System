'''
based on https://www.bilibili.com/video/av29440909
'''

from  sklearn.svm import SVC
import matplotlib.pyplot as plt 
import numpy as np 

class Test_SVM:
    def __init__(self):
        self.X, self.y = self.load_data()
    def load_data(self):
        from sklearn.datasets.samples_generator import make_circles
        X,y = make_circles(100,factor=.1,noise=.1)
        return X,y
    def run_svm(self):
        clf = SVC(kernel='rbf',C=1e6) # kernel:核函数  C:核函数参数，越大映射维度越高
        clf.fit(self.X,self.y)
        # plot
        plt.scatter(self.X[:,0],self.X[:,1],c=self.y,s=50,cmap='autumn')
        plot_svc_decision(clf)
        plt.show()

def plot_svc_decision(model,ax=None,plot_support=True):
    # 画图
    if ax is None:
        ax = plt.gca() 
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y, X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(),Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary margins
    ax.contour(X,Y,P,colors='k',
                levels=[-1,0,1],alpha=0.5,
                linestyles=['--','-','--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],
                    model.support_vectors_[:,1],
                    s=300,linewidth=1,facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
if __name__ == "__main__":
    svm = Test_SVM()
    svm.run_svm()
