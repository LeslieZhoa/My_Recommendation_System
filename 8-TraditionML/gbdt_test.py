from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics 
from sklearn.metrics import mean_squared_error 
import pandas as pd 
import os  
from sklearn import tree
import graphviz
from sklearn.externals.six import StringIO
import pydotplus
import random

class ChurnPreGBDT:
    def __init__(self):
        self.file = 'data/telecom-churn-prediction-data.csv'
        self.data = self.feature_transform()
        self.train, self.test = self.split_data()

    def isNone(self,value):
        # 空缺值用0填充
        if value == ' ' or value is None:
            return '0.0'
        else:
            return value
    
    def feature_transform(self,path ='data/new_churn.csv' ):
        # 将字符特征转换成0，1，2....
        if not os.path.exists(path):
            print('start feature transform...')

            # 将字符特征转换成0，1，2....
            feature_dict = {
                'gender':{'Male':'1','Female':'0'},
                'Partner':{'Yes':'1','No':'0'},
                'Dependents':{'Yes':'1','No':'0'},
                'PhoneService':{'Yes':'1','No':'0'},
                'MultipleLines':{'Yes':'1','No':'0','No phone service':'2'},
                'InternetService':{'DSL':'1','Fiber optic':'2','No':'0'},
                'OnlineSecurity':{'Yes':'1','No':'0','No internet service':'2'},
                'OnlineBackup':{'Yes':'1','No':'0','No internet service':'2'},
                'DeviceProtection':{'Yes':'1','No':'0','No internet service':'2'},
                'TechSupport':{'Yes':'1','No':'0','No internet service':'2'},
                'StreamingTV':{'Yes':'1','No':'0','No internet service':'2'},
                'StreamingMovies':{'Yes':'1','No':'0','No internet service':'2'},
                'Contract':{'Month-to-month':'0','One year':'1','Two year':'2'},
                'PaperlessBilling':{'Yes':'1','No':'0'},
                'PaymentMethod':{'Electronic check':'0','Mailed check':'1','Bank transfer (automatic)':'2','Credit card (automatic)':'3',},
                'Churn':{'Yes':'1','No':'0'}
            }
            fw = open(path,'w')
            fw.write('customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,'
                        'InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,'
                        'StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n')
            for line in open(self.file,'r').readlines():
                if line.startswith('customerID'):
                    continue
                customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,\
                InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,\
                StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn \
                    = line.strip().split(',')
                _list = list()
                _list.append(customerID)
                _list.append(self.isNone(feature_dict['gender'][gender]))
                _list.append(self.isNone(SeniorCitizen))
                _list.append(self.isNone(feature_dict['Partner'][Partner]))
                _list.append(self.isNone(feature_dict['Dependents'][Dependents]))
                _list.append(self.isNone(tenure))
                _list.append(self.isNone(feature_dict['PhoneService'][PhoneService]))
                _list.append(self.isNone(feature_dict['MultipleLines'][MultipleLines]))
                _list.append(self.isNone(feature_dict['InternetService'][InternetService]))
                _list.append(self.isNone(feature_dict['OnlineSecurity'][OnlineSecurity]))
                _list.append(self.isNone(feature_dict['OnlineBackup'][OnlineBackup]))
                _list.append(self.isNone(feature_dict['DeviceProtection'][DeviceProtection]))
                _list.append(self.isNone(feature_dict['TechSupport'][TechSupport]))
                _list.append(self.isNone(feature_dict['StreamingTV'][StreamingTV]))
                _list.append(self.isNone(feature_dict['StreamingMovies'][StreamingMovies]))
                _list.append(self.isNone(feature_dict['Contract'][Contract]))
                _list.append(self.isNone(feature_dict['PaperlessBilling'][PaperlessBilling]))
                _list.append(self.isNone(feature_dict['PaymentMethod'][PaymentMethod]))
                _list.append(self.isNone(MonthlyCharges))
                _list.append(self.isNone(TotalCharges))
                _list.append(feature_dict['Churn'][Churn])
                fw.write(','.join(_list))
                fw.write('\n')
            return pd.read_csv(path)
        else:
            return pd.read_csv(path)
    def split_data(self):
        # 划分训练集和测试集
        train, test = train_test_split(
            self.data,test_size=0.1,random_state=40
        )
        return train, test

    def train_model(self):
        # 训练模型
        print('start train model')
        label = 'Churn'
        ID = 'customerID'
        # 取除label和id之外的特征用于搭建决策树
        x_columns = [x for x in self.train.columns if x not in [label,ID]]
        x_train = self.train[x_columns]
        y_train = self.train[label]
        gbdt = GradientBoostingClassifier(
            learning_rate=0.1,n_estimators=200,max_depth=5
        )
        gbdt.fit(x_train,y_train)
        import pickle #pickle模块

        #保存Model
        with open('data/gbdt_model', 'wb') as f:
            pickle.dump(gbdt, f)

        return gbdt 

    def evaluate(self,gbdt):
        # 评估模型
        label = 'Churn'
        ID = 'customerID'
        # 取除label和id之外的特征用于搭建决策树
        x_columns = [x for x in self.test.columns if x not in [label,ID]]
        x_test = self.test[x_columns]
        y_test = self.test[label]
        y_pred = gbdt.predict_proba(x_test)
        new_y_pred = list()
        for y in y_pred:
            # y[0] label=0的概率 y[1] label=1的概率
            new_y_pred.append(1 if y[1]>0.5 else 0)
        mse = mean_squared_error(y_test,new_y_pred)
        accuracy = metrics.accuracy_score(y_test.values,new_y_pred)
        auc = metrics.roc_auc_score(y_test.values,new_y_pred)
        print('mse: %.4f | accuracy: %.4g | auc score: %.4g'%(mse,accuracy,auc))

    def plot_gbdt(self,gbdt):
        # 可视化一颗树
        dot_data=StringIO()
        features = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines',
                        'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                        'StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']
        sample_index = random.randint(0,len(gbdt.estimators_))
        tree.export_graphviz(gbdt[sample_index].item(),out_file=dot_data
                                , feature_names=features
                                , class_names=["yes", "no"]
                                , filled=True
                                , rounded=True)
 
        graph = graphviz.Source(dot_data.getvalue())
        graph.view()

if __name__ == "__main__":
    test_gbdt = ChurnPreGBDT()
    model = test_gbdt.train_model()
    test_gbdt.evaluate(model)
    test_gbdt.plot_gbdt(model)
        

