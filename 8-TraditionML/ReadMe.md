### 传统机器学习算法
- CF等传统方法局限性：<br>
  对于海量数据，稀疏性，实时性难以处理
- #### SVM
  - 思想：试图找一个超平面来对样本进行分割，把正例和反例用超平面分开，并尽可能使正例和反例之间间隔最大
  - 推导:<br>
   函数间隔：<img src="http://latex.codecogs.com/gif.latex?%5Chat%20r_i%20%3D%20y_i%28wx_i&plus;b%29"/><br>
   集合间隔：<img src="http://latex.codecogs.com/gif.latex?r_i%20%3D%20y_i%28%5Cfrac%7Bw%7D%7B%7C%7Cw%7C%7C%7Dx_i&plus;%5Cfrac%7Bb%7D%7B%7C%7Cw%7C%7C%7D%29"/><br>
   所以可以得到svm等最优化方程：<img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20%5Cmax_%7Bw%2Cb%7D%20%5Cfrac%7B%5Chat%20r%7D%7B%7C%7Cw%7C%7C%7D%5C%5C%5B2ex%5D%20s.t.%20%5C%20%5C%20y_i%28wx_i&plus;b%29%20%5Cgeq%20%5Chat%20r%20%5C%5C%5B2ex%5D%20%5Cend%7Barray%7D%20%5Cright."/><br>
其中<img src="http://latex.codecogs.com/gif.latex?%5Chat%20r"/>取值并不影响最优化问题求解，所以上述问题可以等价：<img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20%5Cmin_%7Bw%2Cb%7D%20%5Cfrac%7B1%7D%7B2%7D%7C%7Cw%7C%7C%5E2%5C%5C%5B2ex%5D%20s.t.%20%5C%20%5C%20y_i%28wx_i&plus;b%29-1%20%5Cgeq%200%20%5C%5C%5B2ex%5D%20%5Cend%7Barray%7D%20%5Cright."/><br>
引入松弛变量<img src="http://latex.codecogs.com/gif.latex?y_i%28wx_i&plus;b%29%20%5Cgeq%201-%5Cxi_i"/>,原问题等价为：<img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20%5Cmin_%7Bw%2Cb%7D%20%5Cfrac%7B1%7D%7B2%7D%7C%7Cw%7C%7C%5E2&plus;C%5Csum_%7Bi%3D1%7D%5EN%5Cxi_i%5C%5C%5B2ex%5D%20s.t.%20%5C%20%5C%20y_i%28wx_i&plus;b%29%20%5Cgeq%201-%5Cxi_i%20%5C%5C%5B2ex%5D%20%5Cxi_i%20%5Cgeq%200%20%5Cend%7Barray%7D%20%5Cright."/><br>
其中C越大，<img src="http://latex.codecogs.com/gif.latex?%5Cxi_i%24"/>越小，分类越严格<br>
引入拉格朗日松弛因子：<img src="http://latex.codecogs.com/gif.latex?L%28w%2Cb%2C%5Cxi%2C%5Calpha%2C%5Cmu%29%3D%5Cfrac%7B1%7D%7B2%7D%7C%7Cw%7C%7C%5E2&plus;C%5Csum_%7Bi%3D1%7D%5EN%5Cxi_i-%5Csum_%7Bi%3D1%7D%5EN%5Calpha_i%28y_i%28wx_i&plus;b%29-1&plus;%5Cxi_i%29-%5Csum_%7Bi%3D1%7D%5EN%5Cmu_i%5Cxi_i"/><br>
求导转换可得其对偶问题<img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20%5Cmin_%7Bw%2Cb%7D%20%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bi%3D1%7D%5EN%5Csum_%7Bi%3D1%7D%5EN%5Calpha_i%5Calpha_jy_iy_j%28x_i%5Ccdot%20x_j%29-%5Csum_%7Bi%3D1%7D%5EN%5Calpha_i%5C%5C%5B2ex%5D%20s.t.%20%5C%20%5C%20%5Csum_%7Bi%3D1%7D%5EN%5Calpha_iy_i%3D0%20%5C%5C%5B2ex%5D%200%5Cleq%20%5Calpha_i%20%5Cleq%20C%20%5Cend%7Barray%7D%20%5Cright."/><br>
KKT条件推导：
    - 加入拉格朗日算子同时需满足的条件：<img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20y_i%28wx_i&plus;b%29-1&plus;%5Cxi_i%20%5Cgeq%200%5C%5C%5B2ex%5D%20%5Cxi_i%20%5Cgeq%200%20%5C%5C%5B2ex%5D%20%5Calpha_i%20%5Cgeq%200%20%5C%5C%5B2ex%5D%20%5Cmu_i%20%5Cgeq%200%20%5C%5C%5B2ex%5D%20%5Calpha_i%28y_i%28wx_i&plus;b%29-1&plus;%5Cxi_i%29%3D0%20%5C%5C%5B2ex%5D%20%5Cmu_i%5Cxi_i%3D0%20%5C%5C%5B2ex%5D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20w%7D%3D%20w-%5Csum_%7Bi%3D1%7D%5EN%5Calpha_i%20y_ix_i%3D0%20%5C%5C%5B2ex%5D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cxi%7D%3D%20C-%5Calpha-%5Cmu%3D0%20%5C%5C%5B2ex%5D%20%5Cfrac%7B%5Cpartial%20L%7D%7B%5Cpartial%20%5Cmu%7D%3D%20%5Csum_%7Bi%3D1%7D%5EN%5Calpha_iy_i%3D0%20%5Cend%7Barray%7D%20%5Cright."/><br>
    1. <img src="http://latex.codecogs.com/gif.latex?%5Calpha_i%3D0"/>,<img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20%5Cmu%20%3D%20C-%5Calpha%20%5Cneq%200%20%5C%5C%5B2ex%5D%20%5Cmu_i%5Cxi_i%3D0%20%5C%5C%5B2ex%5D%20%5CRightarrow%20%5Cxi_i%3D0%20%5C%5C%5B2ex%5D%20%5CRightarrow%20y_i%28wx_i&plus;b%29%20%5Cgeq%201%20%5Cend%7Barray%7D%20%5Cright."/>样本到超平面距离不小于1<br>
    2. <img src="http://latex.codecogs.com/gif.latex?0%20%3C%5Calpha%3CC"/>,<img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20%5Calpha%20%5Cneq%200%20%5C%5C%5B2ex%5D%20%5Cmu%20%3D%20C-%5Calpha%20%5Cneq%200%20%5C%5C%5B2ex%5D%20%5Cmu_i%5Cxi_i%3D0%20%5C%5C%5B2ex%5D%20%5CRightarrow%20%5Cxi_i%3D0%20%5C%5C%5B2ex%5D%20%5Calpha_i%28y_i%28wx_i&plus;b%29-1&plus;%5Cxi_i%29%3D0%20%5C%5C%5B2ex%5D%20%5CRightarrow%20y_i%28wx_i&plus;b%29-1%3D0%20%5Cend%7Barray%7D%20%5Cright."/>样本到超平面距离等于1说明样本是支持向量<br>
    3. <img src="http://latex.codecogs.com/gif.latex?%5Calpha%3DC"/>,<img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20%5Cmu%20%3D%20C-%5Calpha%20%3D%200%20%5C%5C%5B2ex%5D%20%5Cmu_i%5Cxi_i%3D0%20%5C%5C%5B2ex%5D%20%5CRightarrow%20%5Cxi_i%20%5Cgeq%200%20%5C%5C%5B2ex%5D%20%5Calpha_i%28y_i%28wx_i&plus;b%29-1&plus;%5Cxi_i%29%3D0%20%5C%5C%5B2ex%5D%20%5CRightarrow%20y_i%28wx_i&plus;b%29-1&plus;%5Cxi_i%3D0%20%5C%5C%5B2ex%5D%20%5CRightarrow%20y_i%28wx_i&plus;b%29%20%5Cleq%201%20%5Cend%7Barray%7D%20%5Cright."/>样本到超平面距离若小于1说明是误分类样本<br>
  - 求解方法SMO:
    1. 取初始值<img src="http://latex.codecogs.com/gif.latex?%5Calpha%5E%7B%280%29%7D%3D0"/>,令k=0
    2. 选取优化变量<img src="http://latex.codecogs.com/gif.latex?%5Calpha_1%5E%7B%28k%29%7D%20%5C%20%5C%20%5Calpha_2%5E%7B%28k%29%7D"/>,解析求解两个变量优化问题，求解最优解<img src="http://latex.codecogs.com/gif.latex?%5Calpha_1%5E%7B%28k&plus;1%29%7D%20%5C%20%5C%20%5Calpha_2%5E%7B%28k&plus;1%29%7D"/>,更新<img src="http://latex.codecogs.com/gif.latex?%5Calpha"/>为<img src="http://latex.codecogs.com/gif.latex?%5Calpha%5E%7B%28k&plus;1%29%7D"/>
    3. 在精度<img src="http://latex.codecogs.com/gif.latex?%5Cxi"/>范围内满足停机条件时转d，否则k=k+1,转b<br>
       停机条件：<img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20%5Csum_%7Bi%3D1%7D%5EN%5Calpha_iy_i%3D0%20%5C%5C%5B2ex%5D%200%20%5Cleq%20%5Calpha_i%20%5Cleq%20C%20%5Cend%7Barray%7D%20%5Cright."/><br>
KKT:：<img src="http://latex.codecogs.com/gif.latex?%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20y_ig%28x_i%29%20%3D%20%5Cbegin%7Bcases%7D%20%5Cgeq%201%20%26%20%5C%7Bx%7C%5Calpha_i%3D0%5C%7D%20%5C%5C%5B2ex%5D%20%3D1%20%26%20%5C%7Bx_i%7C0%3C%5Calpha_i%3CC%5C%7D%20%5C%5C%5B2ex%5D%20%5Cgeq%201%20%26%20%5C%7Bx_i%7C%5Calpha%3DC%5C%7D%20%5Cend%7Bcases%7D%20%5C%5C%5B4ex%5D%20g%28x_i%29%3D%5Csum_%7Bj%3D1%7D%5EN%5Calpha_jy_jk%28x_j%2Cx_i%29&plus;b%20%5Cend%7Barray%7D%20%5Cright."/><br>
    4. 取<img src="http://latex.codecogs.com/gif.latex?%5Chat%20%5Calpha%3D%5Calpha%5E%7B%28k&plus;1%29%7D"/>
  - 核函数：<br>
  原始空间线性不可分，可以使用一个非线性映射将原始数据变换到另一个高维特征空间，使之线性可分
    - 多项式核<img src="http://latex.codecogs.com/gif.latex?k%28x%2Cz%29%3D%28x%20%5Ccdot%20z&plus;1%29%5Ep"/>-->特征维数高或样本数量多
    - 高斯核<img src="http://latex.codecogs.com/gif.latex?k%28x%2Cz%29%3D%5Cexp%28%5Cfrac%7B-%7C%7Cx-z%7C%7C%5E2%7D%7B2%5Csigma%5E2%7D%29"/>-->样本数量客观，特征少
  - 损失函数hinge loss --><img src="http://latex.codecogs.com/gif.latex?L%3D%5Cmax%280%2C1-y%5Ccdot%20%5Chat%20y%29"/>
  - 优点：只用小样本知识向量就可以解决问题，避免维数灾
  - 缺点：大规模样本困难
- Boosting:
  - 步骤：
  ```md
  * 初始化相同权重样本学习弱学习器
  * 再用弱学习器学习误差来更新样本权重和学习器比重
  * 用新权重继续学习弱学习器，以此重复
  * 将若干学习器累加，得最终学习器
  ```
  - 举例：AdaBoost思路完全一致
  - 特点：最小化损失，bias自然下降，各子模型强相关，不能显著降低variance
- Bagging:
  - 步骤：
  ```md
  * 有放回取样本
  * 根据样本计算统计量
  * 重复N次，得到N个统计量T
  * 由N个统计量计算统计量置信区间
  ```
  - 特点：单个子模型接近不能显著降低bias，各子模型独立可显著降低variance
  - 举例：随机森林RF
    - 步骤：
    ```md
    * 从样本有放回采样n个样本
    * 从所有属性随机选择k个属性，选择最佳分割属性建立决策树
    * 重复以上两步m次，建立m棵决策树
    * 这m棵决策树形成随机森林，通过投票表决结果
    ```
    - 优点：
      1. 可处理高纬度数据
      2. 泛化能力强
      3. 可并行，训练速度快
      4. 对不平衡数据集可平衡误差
      5. 若有特征缺失也可维持准确度
    - 缺点：
      1. 噪声较大，数据会过拟合
      2. 小数据，少特征效果可能差
- 基尼指数，熵，分类误差率关系：
  1. 二分类基尼系数：<img src="http://latex.codecogs.com/gif.latex?Gini%28p%29%3D2p%281-p%29"/>
  2. 熵：<img src="http://latex.codecogs.com/gif.latex?H%28x%29%3D-%5Csum_%7Bi%3D1%7D%5Enp_i%5Clog%20p_i"/>
  3. 分类误差率：<img src="http://latex.codecogs.com/gif.latex?L%20%3D%20%5Cbegin%7Bcases%7D%20p%20%26%20p%20%3C%200.5%20%5C%5C%5B2ex%5D%201-p%20%26%20otherwise%20%5Cend%7Bcases%7D"/><br><div align=center><img src="https://img-blog.csdnimg.cn/20190712150936222.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2E4NTc1NTMzMTU=,size_16,color_FFFFFF,t_70"/></div>
- #### GBDT
  - 原理：<br>
  <img src="http://latex.codecogs.com/gif.latex?L_n%28y%2C%5Chat%20y%5E%7B%28n%29%7D%29%3D%5Csum_%7Bi%3D1%7D%5Eml%28y_i%2C%5Chat%20y_i%5E%7B%28n%29%7D%29%3D%5Csum_%7Bi%3D1%7D%5Enl%28y_i%2C%5Chat%20y_i%5E%7B%28n-1%29%7D&plus;f_n%28x_i%29%29"/><br>
一级导数泰勒展开<img src="http://latex.codecogs.com/gif.latex?l%28y_i%2C%5Chat%20y_i%5E%7B%28n-1%29%7D&plus;f_n%28x_i%29%29%3Dl%28y_i%2C%5Chat%20y_i%5E%7B%28n-1%29%7D%29&plus;%5Cfrac%7B%5Cpartial%20l%28y_i%2C%5Chat%20y_i%5E%7B%28n-1%29%7D%29%7D%7B%5Cpartial%20%5Chat%20y_i%5E%7B%28n-1%29%7D%7Df_n%28x_i%29"/><br>
若<img src="http://latex.codecogs.com/gif.latex?f_n%28x_i%29%3D-%5Cfrac%7B%5Cpartial%20l%28y_i%2C%5Chat%20y_i%5E%7B%28n-1%29%7D%29%7D%7B%5Cpartial%20%5Chat%20y_i%5E%7B%28n-1%29%7D%7D"/>--><img src="http://latex.codecogs.com/gif.latex?l%28y_i%2C%5Chat%20y_i%5E%7B%28n%29%7D%29%3Dl%28y_i%2C%5Chat%20y_i%5E%7B%28n-1%29%7D%29-f_n%5E2%28x_i%29%20%5Cleq%20l%28y_i%2C%5Chat%20y_i%5E%7B%28n-1%29%7D%29"/><br>
所以向梯度方向更新，效果会越来越好
  - 步骤：
    1. 初始化函数F0常量<img src="http://latex.codecogs.com/gif.latex?F_0%28x%29%3Darg%20%5Cmin_%7B%5Cgamma%7D%5Csum_%7Bi%3D1%7D%5EnL%28y_i%2C%5Cgamma%29"/>
    2. 计算第m棵树对应梯度<img src="http://latex.codecogs.com/gif.latex?r_%7Bm%2Ci%7D%3D-%5B%5Cfrac%7B%5Cpartial%20L%28y_i%2CF%28x_i%29%29%7D%7B%5Cpartial%20F%28x%29%7D%5D_%7BF%28x%29%3DF_%7Bm-1%7D%28x%29%7D"/>
    3. 使用CART回归树拟合<img src="http://latex.codecogs.com/gif.latex?%5C%7B%28x_1%2Cr_%7Bm%2C1%7D%29%2C%28x_2%2Cr_%7Bm%2C2%7D%29%2C...%2C%28x_n%2Cr_%7Bm%2Cn%7D%29%5C%7D"/>得到第m棵树叶子节点区域<img src="http://latex.codecogs.com/gif.latex?R_%7Bj%2Cm%7D%20%5C%20j%5Cin%201%2C2%2C...%2CJ_m"/>
    4. 根据上述分完叶节点计算最优权重<img src="http://latex.codecogs.com/gif.latex?r_%7Bj%2Cm%7D%3Darg%20%5Cmin_%7B%5Cgamma%7D%5Csum_%7Bx%5Cin%20R_%7Bj%2Cm%7D%7DL%28y_i%2CF_%7Bm-1%7D%28x_i%29&plus;%5Cgamma%29"/>
    5. 更新<img src="http://latex.codecogs.com/gif.latex?F_m%28x%29%3DF_%7Bm-1%7D%28x%29&plus;%5Calpha%20%5Csum_%7Bj%3D1%7D%5E%7BJ_m%7D%5Cgamma_%7Bj%2Cm%7DI%28x%20%5Cin%20R_%7Bj%2Cm%7D%29"/>其中<img src="http://latex.codecogs.com/gif.latex?%5Calpha"/>为学习率
    6. 重复b->e得到最终累加树
  - 可限制参数：
    1. 每棵树深度，叶节点数，分支样本数
    2. 学习率
    3. 迭代次数
  - 特点：GBDT必须是回归树，因为存在一阶导要连续
  - 优点：
    1. 残差累加防止过拟合
    2. 无需复杂特征工程
    3. 可解释性强
  - 缺点：
    1. 不好并行
    2. 不适合高维度稀疏特征
- #### XGBoost
  - 推导：<br>
    <img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20obj%5Et%20%26%3D%5Csum_%7Bi%3D1%7D%5Enl%28y_i%2C%5Chat%20y_i%5E%7B%28t%29%7D%29&plus;%5Csum_%7Bi%3D1%7D%5Et%5COmega%20%28f_i%29%20%5C%5C%5B2ex%5D%20%26%3D%5Csum_%7Bi%3D1%7D%5Enl%28y_i%2C%5Chat%20y_i%5E%7B%28t-1%29%7D&plus;f_t%28x_i%29%29&plus;%5COmega%20%28f_t%29&plus;C%20%5Cend%7Baligned%7D"/>其中<img src="http://latex.codecogs.com/gif.latex?%5COmega"/>为正则项，C为常数<br>
二阶泰勒展开:
    - <img src="http://latex.codecogs.com/gif.latex?g_i%3D%5Cpartial%20%5Chat%20y%5E%7B%28t-1%29%7Dl%28y_i%2C%5Chat%20y%5E%7B%28t-1%29%7D%29%20%5C%20%5C%20%5C%20h_i%3D%5Cpartial%5E2%5Chat%20y%5E%7B%28t-1%29%7Dl%28y_i%2C%5Chat%20y%5E%7B%28t-1%29%7D%29"/>
    - <img src="http://latex.codecogs.com/gif.latex?obj%5E%7B%28t%29%7D%5Capprox%20%5Csum_%7Bi%3D1%7D%5En%5Bl%28y_i%2C%5Chat%20y_i%5E%7B%28t-1%29%7D%29&plus;g_if_t%28x_i%29&plus;%5Cfrac%7B1%7D%7B2%7Dh_if_t%5E2%28x_i%29%5D&plus;%5COmega%20%28f_t%29&plus;C"/><br>
    其中<img src="http://latex.codecogs.com/gif.latex?l%28y_i%2C%5Chat%20y_i%5E%7B%28t-1%29%7D%29"/>为常数项
    - <img src="http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20new%5C_%20obj%5E%7B%28t%29%7D%20%26%3D%5Csum_%7Bi%3D1%7D%5En%5Bg_if_t%28x_i%29&plus;%5Cfrac%7B1%7D%7B2%7Dh_if_t%5E2%28x&plus;i%29%5D&plus;%5COmega%20%28f_t%29%20%5C%5C%5B2ex%5D%20%26%3D%20%5Csum_%7Bi%3D1%7D%5En%5Bg_iw_q%28x_i%29&plus;%5Cfrac%7B1%7D%7B2%7Dh_iw_q%5E2%28x&plus;i%29%5D&plus;%5Cgamma%20T&plus;%5Clambda%20%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bj%3D1%7D%5ETw_j%5E2%20%5C%5C%5B2ex%5D%20%26%3D%20%5Csum_%7Bj%3D1%7D%5ET%5B%28%5Csum_%7Bi%5Cin%20I_j%7Dg_i%29w_j&plus;%5Cfrac%7B1%7D%7B2%7D%28%5Csum_%7Bi%5Cin%20I_j%7Dh_i&plus;%5Clambda%29w_j%5E2%5D&plus;%5Clambda%20T%20%5Cend%7Baligned%7D"/>
    - 令<img src="http://latex.codecogs.com/gif.latex?G_j%3D%5Csum_%7Bi%5Cin%20I_j%7Dg_i%20%5C%20%5C%20%5C%20H_j%3D%5Csum_%7Bi%5Cin%20I_j%7Dh_i%20%5C%5C%5B2ex%5D%20obj%5E%7B%28t%29%7D%3D%5Csum_%7Bj%3D1%7D%5ET%5BG_jW_%5Cfrac%7B1%7D%7B2%7D%28H_j&plus;%5Clambda%29W_j%5E2%5D&plus;%5Cgamma%20T"/>
    - 导数为0，<img src="http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20J%28f_t%29%7D%7B%5Cpartial%20W_j%7D%3DG_j&plus;%28H_j&plus;%5Clambda%20%29W_j%3D0%20%5C%5C%5B2ex%5D%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20%5C%20W_j%3D-%5Cfrac%7BG_j%7D%7BH_j&plus;%5Clambda%7D%20%5C%20%5C%20%5C%20obj%3D-%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bj%3D1%7D%5ET%5Cfrac%7BG_j%5E2%7D%7BH_j&plus;%5Clambda%7D&plus;%5Cgamma%20T"/><br>
    - 新的分割点选择标准：<img src="http://latex.codecogs.com/gif.latex?Gain%3D%5Cfrac%7B1%7D%7B2%7D%5B%5Cfrac%7BG_L%5E2%7D%7BH_L&plus;%5Clambda%7D&plus;%5Cfrac%7BG_R%5E2%7D%7BH_R&plus;%5Clambda%7D&plus;%5Cfrac%7B%28G_L&plus;G_R%29%5E2%7D%7BH_L&plus;H_R&plus;%5Clambda%7D%5D"/>
  - 步骤：
    1. 初始化树
    2. 计算Gj,Hj，由Gain找分支建立树
    3. 重复b建立m棵树累加
  - 优点：
    1. 精度高，采用二阶泰勒展开，增加精度，也可自定义损失函数
    2. 防止过拟合，加入正则项
    3. 可以处理缺失值
    4. 支持并行：确定分割点，预先对数据进行排序，保存为block结构，后面重复使用，减小计算量，实现并行
  - 缺点：复杂度高
- #### GBDT+LR
  - 思路：<br>
  ```md
  * 对特征进行GBDT优化
  * 把所有树的叶节点当作one-hot特征，对应计算数据的gbdt特征
  * gbdt特征与原始数据特征一起加入LR训练
  ```

