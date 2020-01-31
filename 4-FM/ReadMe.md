## 因式分解机FM<br>
*多用于ctr问题*
### 写在前面
此问题是判断用户是否点击，one-hot特征怎么搭建呢？<br>
假设只有两种特征hour和app，hour有1-24共24种选择，app有A,B两种选择，那么建立的one-hot特征就是26维
### 怎么运行代码
- 下载数据[此链接](https://www.kaggle.com/c/avazu-ctr-prediction/data),下载并解压数据，将最终解压的train重命名为train.csv放置到该目录下的avazu-ctr-prediction目录下，该文件夹是自己新建的
- 运行代码 
  - 运行FM python FM.py 
  - 运行FFM python FFM.py
  - 运行DeepFM python DeepFM.py
  - *里面的一些参数设置可以自己去慢慢鼓捣*
### 一些原理
- FM
  - 表达方式：<img src="http://latex.codecogs.com/gif.latex?y%3Dw_0&plus;%5Csum_%7Bi%3D1%7D%5Enw_ix_i&plus;%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3Di&plus;1%7D%5En%3CV_i%2CV_j%3Ex_ix_j"/><br> 化简一下可得到：<img src="http://latex.codecogs.com/gif.latex?%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3Di&plus;1%7D%5En%3CV_i%2CV_j%3Ex_ix_j%20%3D%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bf%3D1%7D%5Ek%28%28%5Csum_%7Bi%3D1%7D%5Env_%7Bif%7Dx_i%29%5E2-%5Csum_%7Bi%3D1%7D%5Env_%7Bif%7D%5E2x_i%5E2%29"/>
  - 特点：相较于LR考虑到特征之间的关联，对权重简化，有利于减小计算复杂度
- FFM
  - 表达式：<img src="http://latex.codecogs.com/gif.latex?y%3Dw_0&plus;%5Csum_%7Bi%3D1%7D%5Enw_ix_i&plus;%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3Di&plus;1%7D%5En%3CV_%7Bi%2Cf_j%7D%2CV_%7Bj%2Cf_i%7D%3Ex_ix_j"/>
  - 特点：相较于FM，每一维特征针对它对每一种field<img src="http://latex.codecogs.com/gif.latex?f_j"/>都会学习一个隐向量<img src="http://latex.codecogs.com/gif.latex?V_%7Bi%2Cf_j%7D"/>，从而更好的关联特征之间的关系。但相应计算复杂度随之增加。
  - 解释field，假设有个特征是hour，一共有[1,2,3,...,24]一共24个可选择，我们把这24个选择算作24个维度的特征（用one-hot表示），那这24个维度就同属同一个field。
- DeepFFM
  - 表达式：<img src="http://latex.codecogs.com/gif.latex?y_%7Bfm%7D%3Dw_0&plus;%5Csum_%7Bi%3D1%7D%5Enw_ix_i&plus;%5Csum_%7Bi%3D1%7D%5En%5Csum_%7Bj%3Di&plus;1%7D%5En%3CV_i%2CV_j%3Ex_ix_j"/><br>
    将权重V&#8194;reshape成[-1,f*k]（依然如ffm引入了隐式特征）并接入DNN中<img src="http://latex.codecogs.com/gif.latex?y_%7Bdnn%7D%3DDNN%28V_%7Breshape%7D%29"/><br>
    &#8194;&#8194;&#8194;&#8194;&#8194;<img src="http://latex.codecogs.com/gif.latex?y_%7Bout%7D%3Dy_%7Bfm%7D&plus;y_%7Bdnn%7D"/>
    
  - 特点：从原始数据中同时学习到了低维与高维特征；
不再需要特征工程