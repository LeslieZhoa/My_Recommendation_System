## 逻辑回归LR
### 写在前面
此问题是判断用户是否点击，one-hot特征怎么搭建呢？<br>
假设只有两种特征hour和app，hour有1-24共24种选择，app有A,B两种选择，那么建立的one-hot特征就是26维
### 怎么运行代码
- 下载数据[此链接](https://www.kaggle.com/c/avazu-ctr-prediction/data),下载并解压数据，将最终解压的train重命名为train.csv放置到该目录下的avazu-ctr-prediction目录下，该文件夹是自己新建的
- 运行代码 python LR.py  里面的一些参数设置可以自己去慢慢鼓捣
### 一些原理
- 表示形式
    <div align=center><img src="http://latex.codecogs.com/gif.latex?%5C%5C%20A_0%20%3D%20x%20%5C%5C%20Z_i%3DW_iA_%7Bi-1%7D%20%5C%5C%20A_i%3Dsigmoid%28Z_i%29%20%5C%5C%20y%20%3D%20A_n%20%5C%5C%20loss%20%3D%20cross%5C_entropy"/></div>
- 为什么激活函数使用sigmoid
  - LR假设函数概率服从伯努利分布，写成指数族分布形式可以数学推导出激活函数为sigmiod
  - 对于二分类问题，假设第i个特征对应第k类第贡献是<img src="http://latex.codecogs.com/gif.latex?w_%7Bki%7D"/>,则数据点<img src="http://latex.codecogs.com/gif.latex?%28x_1%2Cx_2%2C...%2Cx_n%29"/>属于第k类第概率正比与<img src="http://latex.codecogs.com/gif.latex?%5Cexp%28%7Bw_%7Bk1%7Dx_1&plus;...&plus;w_%7Bkn%7Dx_n%7D%29"/> <br>
    因为一个数据点属于各类第概率之和为1，所以
    <img src="http://latex.codecogs.com/gif.latex?p%28y%3D1%29%3D%5Cfrac%7B%5Cexp%28%5Csum_%7Bi%3D1%7D%5Enw_%7B1i%7Dx_i%29%7D%7B%5Cexp%28%5Csum_%7Bi%3D1%7D%5Enw_%7B1i%7Dx_i%29&plus;%5Cexp%28%5Csum_%7Bi%3D1%7D%5Enw_%7B0i%7Dx_i%29%7D"/>
       分子分母同时除以分子，设<img src="http://latex.codecogs.com/gif.latex?w_i%3Dw_%7B1i%7D-w_%7B0i%7D"/>则<img src="http://latex.codecogs.com/gif.latex?%28y%3D1%29%3D%5Cfrac%7B1%7D%7B1&plus;%5Cexp%28-%5Csum_%7Bi%3D1%7D%5Enw_ix_i%29%7D"/>即为sigmoid函数
- 为什么分类问题损失函数不使用mse
  - mse损失函数为<img src="http://latex.codecogs.com/gif.latex?L%3D%5Cfrac%7B%28y-y%27%29%5E2%7D%7B2%7D"/>导数为<img src="http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w%7D%3D%28y%27-y%29%5Csigma%27%28wx%29x"/>其中<img src="http://latex.codecogs.com/gif.latex?%5Csigma%27%28wx%29%3Dwx%281-wx%29"/>。根据w初始化，导数值可能很小，从而导致收敛变慢，而训练过程中也可能因为该值过小而提早终止训练（梯度消失）。而交叉熵<img src="http://latex.codecogs.com/gif.latex?g%27%3D%5Csum%28x_i%28y_i-p%28x_i%29%29"/>则没有此问题
- L1和L2
  - L1导数为sign(w)， 不管L1大小是多少，只要不是0就是1或者-1，所以每次更新都是稳步向0前进，更容易产生稀疏特征，用于特征选择；对应于拉普拉斯分布，对极端值更容易容忍。
  - L2导数为w，越靠近0，梯度越小，值越靠近0，所以L2产生更多特征接近0；L2对应高斯分布。
- 优缺点
  - 优点：最后输出为概率值，计算简单方便
  - 缺点：假设特征之间相互独立，无法求解多次问题