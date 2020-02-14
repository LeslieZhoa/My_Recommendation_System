from sklearn import metrics 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.linear_model import LogisticRegression 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder 
import os 
import random
import numpy as np
class ChurnPredWithGBLR:
    def __init__(self):
        self.file = 'data/new_churn.csv'
        self.one_hot_data, self.data = self.load_data()
        self.train_lr, self.test_lr, self.train_gb, self.test_gb = self.split() 

    def load_data(self,file='data/one_hot_churn.csv'):
        data = pd.read_csv(self.file)
        # 特征one-hot处理
        if not os.path.exists(file):
            
            labels = list(data.keys())
            fDict = dict()
            length = 0
            for f in labels:
                if f not in ['customerID','tenure','MonthlyCharges','TotalCharges','Churn']: #除去id,label和有数值的特征
                    fDict[f] = sorted(list(data.get(f).unique()))
                    length += len(fDict[f])
            
            # 写入文件
            fw = open(file,'w')
            fw.write('customerID,')
            for i in range(1,length+4):
                fw.write('f_%s,'%i)
            fw.write('Churn\n')
            for line in data.values:
                list_line = list(line)
                list_result = list()
                for i in range(len(list_line)):
                    if labels[i] in ['customerID','tenure','MonthlyCharges','TotalCharges','Churn']:
                        list_result.append(list_line[i])
                    else:
                        # one hot
                        arr = [0] * len(fDict[labels[i]])
                        ind = fDict[labels[i]].index(list_line[i])
                        arr[ind] = 1
                        for one in arr: list_result.append(one)
                fw.write(','.join([str(f) for f in list_result])+'\n')
            fw.close()
        return pd.read_csv(file),pd.read_csv(self.file)

    def split(self):
        test_index = random.sample(range(len(self.data)),round(len(self.data)*0.1))
        train_index = list(set(range(len(self.data))) ^ set(test_index))
        train_lr, test_lr = self.one_hot_data.loc[train_index],self.one_hot_data.loc[test_index]
        train_gb, test_gb = self.data.loc[train_index], self.data.loc[test_index]
        return train_lr,test_lr,train_gb,test_gb
    def train_model(self):
        # 模型训练
        # one-hot 特征
        print("Start Train Model ... ")
        label = "Churn"
        ID = "customerID"
        lr_columns = [x for x in self.train_lr.columns if x not in [label, ID]]
        gb_columns = [x for x in self.train_gb.columns if x not in [label, ID]]
        x_train_lr = self.train_lr[lr_columns]
        y_train_lr = self.train_lr[label]
        # 创建gbdt模型并训练
        x_train_gb = self.train_gb[gb_columns]
        y_train_gb = self.train_gb[label]

        gbdt = GradientBoostingClassifier()
        gbdt.fit(x_train_gb,y_train_gb)
        # 获取 gbdt特征
        enc = OneHotEncoder()
        gbdt_index = gbdt.apply(x_train_gb)
        # 100为n_estimators，决策树个数
        enc.fit(gbdt_index.reshape(-1,100))
        new_x_gb = enc.transform(gbdt_index.reshape(-1,100)).todense()
        
        new_x_train = np.concatenate([x_train_lr,new_x_gb],axis=-1)
        np.save('new_x',new_x_train)
        np.save('new_y',y_train_gb)
        # 训练LR
        lr = LogisticRegression()
        lr.fit(new_x_train,y_train_gb)
        return enc,gbdt,lr

    def evaluate(self,enc,gbdt,lr):
        label = 'Churn'
        ID = 'customerID'
        # one-hot 特征
        lr_columns = [x for x in self.test_lr.columns if x not in [label, ID]]
        gb_columns = [x for x in self.test_gb.columns if x not in [label, ID]]
        x_test_lr = self.test_lr[lr_columns]
        y_test_lr = self.test_lr[label]

        # gbdt 模型评估效果
        x_test_gb = self.test_gb[gb_columns]
        gbdt_y_pred = gbdt.predict_proba(x_test_gb)
        new_gbdt_y_pred = list()
        for y in gbdt_y_pred:
            new_gbdt_y_pred.append(1 if y[1]>0.5 else 0)
        mse_gb = mean_squared_error(y_test_lr,new_gbdt_y_pred)
        acc_gb = metrics.accuracy_score(y_test_lr,new_gbdt_y_pred)
        auc_gb = metrics.roc_auc_score(y_test_lr,new_gbdt_y_pred)
        print('GBDT result: mse: %.4f | accuracy: %.4g | auc: %.4g'%(mse_gb,acc_gb,auc_gb))

        new_x = np.concatenate([x_test_lr,enc.transform(gbdt.apply(x_test_gb).reshape(-1,100)).todense()],axis=-1)
        lr_y_pred = lr.predict_proba(new_x)
        
        new_lr_y_pred = list()
        for y in lr_y_pred:
            new_lr_y_pred.append(1 if y[1]>0.5 else 0)
        mse_lr = mean_squared_error(y_test_lr,new_lr_y_pred)
        acc_lr = metrics.accuracy_score(y_test_lr,new_lr_y_pred)
        auc_lr = metrics.roc_auc_score(y_test_lr,new_lr_y_pred)
        print('Mix result: mse: %.4f | accuracy: %.4g | auc: %.4g'%(mse_lr,acc_lr,auc_lr))
if __name__ == "__main__":
    mix = ChurnPredWithGBLR()
    enc,gbdt,lr = mix.train_model()
    mix.evaluate(enc,gbdt,lr)
               