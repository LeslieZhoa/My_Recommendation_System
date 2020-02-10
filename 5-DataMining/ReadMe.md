### 数据挖掘
- 数据预处理
  - 标准化
    - Min-Max标准化<img src="http://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%7Bx-x_%7Bmin%7D%7D%7Bx_%7Bmax%7D-x_%7Bmin%7D%7D"/>
    - Z-score标准化<img src="http://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%7Bx-%5Cmu%7D%7B%5Csigma%7D"/>
    - 小数定标标准化<img src="http://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%7Bx%7D%7B10%5Ej%7D"/> <img src="http://latex.codecogs.com/gif.latex?j"/> 表示满足条件最小整数，如最大数为997，<img src="http://latex.codecogs.com/gif.latex?j%3D3"/>
    - 均值归一化<img src="http://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%7Bx-%5Cmu%7D%7Bx_%7Bmax%7D-x_%7Bmin%7D%7D"/>
    - 向量归一化<img src="http://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%7Bx%7D%7B%5Csum_%7Bi%3D1%7D%5En%20x_i%7D"/>
    - 指数转换：
      - lg函数<img src="http://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%7Blg%28x%29%7D%7Blg%28x_%7Bmax%7D%29%7D"/>
      - softmax<img src="http://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%7Be%5Ex%7D%7B%5Csum%20e%5E%7Bx_i%7D%7D"/>
      - sigmoid<img src="http://latex.codecogs.com/gif.latex?x%27%3D%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-x%7D%7D"/>
  - 离散化
    - 等宽分组：由分组个数得到固定宽度分组。缺点：区间样本分布不均匀
    - 等频分组：分组后每组变量个数相同。缺点：会将相同变量分到不同组
    - 单变量分组：将所有变量按降序或升序排序，将相同变量分为同一组，排序名次即为排序结果
    - 基于信息熵分组<br>
      步骤：
      1. 对属性A对的所有取值从大到小排序
      2. 遍历属性A的每个值V,将属性A的值分成两个区间S1，S2,使其作为分隔点划分数据集后的熵S最小
      3. 当划分的熵大于设置阈值且小于指定数据分组个数时，递归对S1，S2执行步骤b中对划分
- 数据抽样
  - 随机抽样
  - 分层抽样
  - 系统抽样
  - 渐近抽样：通过掌握模型准确率随样本增大变化情况选取接近于稳定点对其他样本，可以估算出稳定点对接近程度，从而决定是否停止抽样
- 数据降维
  - PCA
    将n维样本点转换成k维后，每维上样本方差都很大
    - 步骤：
      1. 特征标准化
      2. 计算协方差矩阵
      3. 计算协方差矩阵特征值和特征向量
      4. 选取最大k个特征值对应特征向量，得到特征向量矩阵
      5. 将数据与特征向量矩阵相乘变换到k维，得到新的数据集
      6. 提取后数据信息占比<img src="http://latex.codecogs.com/gif.latex?%5Csqrt%7B%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5Er%5Csigma_i%5E2%7D%7B%5Csum_%7Bi%3D1%7D%5Ek%5Csigma_i%5E2%7D%7D"/>其中<img src="http://latex.codecogs.com/gif.latex?%5Csigma_i%3D%5Csqrt%7B%5Clambda_i%7D"/>
    - 优点：
      - 仅需方差衡量信息量，不受数据集以外因素影响
      - 各主成分之间正交，可消除原始数据成分之间相互影响
      - 计算简单，主要运算是特征分解，易于实现
    - 缺点：
      - 主成分各特征含义模糊，不如原始数据可解释性强
      - 方差小对非主成分可能含有对样本差异重要信息
  - T-SNE:<br>
    拟合低维数据，使其分布概率与真实数据分布KL散度最小
    - 假设：
      1. 高维数据点<img src="http://latex.codecogs.com/gif.latex?x_1%2Cx_2%2C...%2Cx_n%2Cp_%7Bi%7Cj%7D"/>默认<img src="http://latex.codecogs.com/gif.latex?p_%7Bi%7Ci%7D%3D0"/>
      2. 低维数据点<img src="http://latex.codecogs.com/gif.latex?y_1%2Cy_2%2C...%2Cy_n"/>
    - 方法：
      1. SNE:
        - 公式：<br>
          <img src="http://latex.codecogs.com/gif.latex?p_%7Bj%7Ci%7D%3D%5Cfrac%7B%5Cexp%28-%7C%7Cx_i-x_j%7C%7C%5E2/2%5Csigma%20_i%5E2%29%7D%7B%5Csum_%7Bk%20%5Cne%20i%7D%20%5Cexp%28-%7C%7Cx_i-x_k%7C%7C%5E2/2%5Csigma%20_i%5E2%7D"/><br>
          <img src="http://latex.codecogs.com/gif.latex?q_%7Bj%7Ci%7D%3D%5Cfrac%7B%5Cexp%28-%7C%7Cy_i-y_j%7C%7C%5E2%29%7D%7B%5Csum_%7Bk%20%5Cne%20i%7D%20%5Cexp%28-%7C%7Cy_i-y_k%7C%7C%5E2%7D"/><br>
          <img src="http://latex.codecogs.com/gif.latex?C%3D%5Csum%20_i%20KL%28P_i%7C%7CQ_i%29%3D%5Csum_i%20%5Csum_j%20p_%7Bj%7Ci%7Dlog%20%5Cfrac%7Bp_%7Bj%7Ci%7D%7D%7Bq_%7Bj%7Ci%7D%7D"/><br>
          <img src="http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20y_i%7D%3D2%5Csum_j%28p_%7Bj%7Ci%7D-q_%7Bj%7Ci%7D&plus;p_%7Bi%7Cj%7D-q_%7Bi%7Cj%7D%29%28y_i-y_j%29"/><br>
        - 参数选择：
          1. 用KL散度衡量不对称，如果高维度相邻而低维度分开(p大q小)，cost很大；如果高纬度数据分开而低维度数据相邻(p小q大)cost很小；SNE倾向于保留高维数据大局部结构
          2. <img src="http://latex.codecogs.com/gif.latex?%5Csigma_i"/>取值<br>
            困惑度<img src="http://latex.codecogs.com/gif.latex?Prep%28p_i%29%3D2%5E%7BH%28p_i%29%7D%20%5C%20%5C%20H%28p_i%29%3D-%5Csum_j%20p_%7Bj%7Ci%7D%20log_2%20p_j%7Ci"/>困惑度通常设置5到50之间<br>
            熵大-->困惑度大-->分布相对平坦，每个元素概率更相近
      2. 对称SNE
        - 公式：<br>
          <img src="http://latex.codecogs.com/gif.latex?p_%7Bij%7D%3D%5Cfrac%7Bp_%7Bi%7Cj%7D&plus;p_%7Bj%7Ci%7D%7D%7B2N%7D"/><br>
            <img src="http://latex.codecogs.com/gif.latex?q_%7Bij%7D%3D%5Cfrac%7B%5Cexp%28-%7C%7Cy_i-y_j%7C%7C%5E2%29%7D%7B%5Csum_%7Bk%20%5Cne%20i%7D%20%5Cexp%28-%7C%7Cy_i-y_k%7C%7C%5E2%7D"/><br>
            <img src="http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20y_i%7D%3D4%5Csum_j%28p_%7Bij%7D-q_%7Bij%7D%29%28y_i-y_j%29"/>
        - 特点：保证<img src="http://latex.codecogs.com/gif.latex?p_%7Bji%7D"/>不会太小，每个点对代价函数都有贡献，梯度简单
      3. t-sne
        - 公式：<br>
          <img src="http://latex.codecogs.com/gif.latex?q_%7Bij%7D%3D%5Cfrac%7B%281&plus;%7C%7Cy_i-y_j%7C%7C%5E2%29%5E%7B-1%7D%7D%7B%5Csum_%7Bk%20%5Cne%20l%7D%281&plus;%7C%7Cy_k-y_l%7C%7C%5E2%29%5E%7B-1%7D%7D"/><br>
          <img src="http://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20C%7D%7B%5Cpartial%20y_i%7D%3D4%5Csum_j%28p_%7Bij%7D-q_%7Bij%7D%29%28y_i-y_j%29%281&plus;%7C%7Cy_i-y_j%7C%7C%5E2%29%5E%7B-1%7D"/>
        - 特点：将低维分布改为t分布，解决拥挤问题，强制保证低维空间<img src="http://latex.codecogs.com/gif.latex?q_%7Bij%7D%3Ep_%7Bij%7D"/>
    - 优点：保持局部结构能力
    - 缺点：复杂度高，具有随机性
- 数据清理
  - 不合格数据修正
  - 缺失值填充
    - 忽略数据
    - 人工填写
    - 使用全局固定值填充
    - 使用属性中心度量进行填充
    - 使用与给定元组属于同一类所有样本的属性均值或中心值填充
    - 使用回归决策树等工具进行推理
  - 噪声值处理
  - 离群点处理
    - 正太分布检测：计算数据集标准差和方差，根据不同置信区间来排除异常值
    - Tukey's Test :<br>
      基于四分位数计算出数据集中的最小估计值和最大估计值的异常值<br>
      Q1-->下四分位数 Q2-->中位数 Q3-->上四分位数<br>
      最小估计值为Q1-k(Q3-Q1)，最大估计值为Q3+k(Q3-Q1)<br>
      当k=1.5时，小于最小估计值和大于最大估计值当中度异常，k=3时为重度异常
  - 基于模型检测
    - 基于聚类异常点检测
    - 基于回归异常点检测
    - 基于邻近度/距离/相似度异常点检测
    - 基于密度异常点检测
- 相似度计算
  - 闵可夫斯基距离：<img src="http://latex.codecogs.com/gif.latex?%5Csum_%7Bi%3D1%7D%5En%28%7Cx_i-y_i%7C%5Ep%29%5E%7B%5Cfrac%7B1%7D%7Bp%7D%7D"/>
    - p=2为欧式距离只能走直线
    - p=1为曼哈顿距离，只能沿划定格子边缘走
    - p趋于无穷大<img src="http://latex.codecogs.com/gif.latex?max_%7Bi%3D1%7D%5En%7Cx_i-y_i%7C"/>结合欧式和曼哈顿<br>
        缺点：受不同属性尺度不同影响，与数据之间分布无关具有局限性
  - 马氏距离：
    - 表示形式：有m各样本向量(x1,x2,...,xm)协方差矩阵为S，<img src="http://latex.codecogs.com/gif.latex?D%28x_i%2Cx_j%29%3D%5Csqrt%7B%28x_i-x_j%29%5ETS%5E-1%28x_i-x_j%29%7D"/>
    - 物理意义：坐标旋转-->使旋转后各维度之间线性无关；数据压缩-->将不同维度上数据压缩成方差都是1当数据集；马氏距离是旋转变换缩放后当欧式距离
    - 缺点：夸大变化微小变量当作用，受协方差矩阵不稳定影响，马氏距离并不一定总能顺序计算出
  - 余弦相似度：
    - 表示形式：<img src="http://latex.codecogs.com/gif.latex?cos%20sim%28x%2Cy%29%3D%5Cfrac%7B%5Csum%20x_iy_i%7D%7B%5Csqrt%7B%5Csum%20x_i%5E2%7D%5Csqrt%7B%5Csum%20y_i%5E2%7D%7D"/>
    - 优点：与幅值无关，只与向量方向有关
    - 缺点：受到向量平移影响
  - 皮尔逊相关系数：
    - 表达形式：<img src="http://latex.codecogs.com/gif.latex?%5Crho_%7BXY%7D%3D%5Cfrac%7BE%28%28X-EX%29%28Y-EY%29%29%7D%7B%5Csqrt%7BD%28X%29%7D%5Csqrt%7BD%28Y%29%7D%7D%20%5C%20%5C%20%5C%20D_%7BXY%7D%3D1-%5Crho_%7BXY%7D"/>
    - 优点：具有平移不变性和尺度不变性，计算出两个向量（维度）相关性
    - 缺点：假设数据服从正太分布
  - 汉名距离：两个等长字符串s1,s2之间，将一个变为另一个所需最小替换次数
  - 杰卡得相似系数：<img src="http://latex.codecogs.com/gif.latex?J%3D%5Cfrac%7BM_%7B11%7D%7D%7BM_%7B01%7D&plus;M_%7B10%7D&plus;M_%7B11%7D%7D"/>，比汉名距离多了权重
- 数据分类：
  - KNN算法分类
    - 算法流程
      1. 计算未知样本和每个训练样本距离distance
      2. 按照距离distance的递增关系排序
      3. 得到距离最小的前k个样本
      4. 统计k最近邻样本中每个类标号出现的次数
      5. 选择出现频率最高的类标号作为未知样本类标号
    - 缺点：受K值影响，K应为奇数
  - 决策树：
    - 算法流程：
      1. 树从代表训练样本的根节点开始
      2. 如果样本都在同一类中，则该节点为树叶，并用该类标记
      3. 否则，算法选择最有分类能力的属性作为决策树的当前节点
      4. 根据当前决策节点属性取值不同，将训练样本数据集分成若干子集，每个取值形成一个分支，有几个取值，就形成几个分支
      5. 针对步骤d得到的每个子集，重复a,b,c递归形成每个划分样本上的决策树，一旦一个属性只出现在一个节点上，就不必在该节点任何子节点考虑它
    - 终止条件：
      1. 给定节点所有样本属于同一类
      2. 没有剩余属性可用进一步划分样本
      3. 某一分支没有满足该分支已有分类样本
    - 剪枝：
      1. 加入随机量
      2. 限制最大深度
      3. 限制节点分支后每个节点包含最小样本
      4. 限制节点包含多少样本才被分支
      5. 限制分支特征个数
      6. 限制信息增益大小
      7. 权重限制
    - 物理意义：
      1. 学习采用自定而下递归方法
      2. 基本思想以信息熵为度量构造一棵下降最快的树，到叶子节点处熵值为0
      3. 每个叶节点中实例都属于同一类
      4. 有监督学习
    - 优点：
      1. 对缺失值不敏感
      2. 可处理不相关特征数据
      3. 效率高，只需构建一次
    - 缺点：
      1. 对连续性字段难以预测
      2. 类别多时计算复杂
      3. 易过拟合
      4. 异常值敏感，泛化能力差
  - 朴素贝叶斯:<br>
    假设各维度特征相互独立
    - 公式：<img src="http://latex.codecogs.com/gif.latex?p%28y_k%7Cx%29%3D%5Cfrac%7Bp%28y_k%29%5Cprod_%7Bi%3D1%7D%5En%20p%28x_i%7Cy_k%29%7D%7B%5Csum_%7Bi%3D1%7D%5Emp%28y_i%29%5Cprod_%7Bi%3D1%7D%5En%20p%28x_i%7Cy_i%29%7D%20%5C%20%5C%20%5C%20f%28x%29%3Dmax%28p%28y_k%7Cx%29%3Dmax%28p%28y_k%29%5Cprod_%7Bi%3D1%7D%5En%20p%28x_i%7Cy_k%29%29"/>
    - 优点：
      1. 有稳定分类效率
      2. 对小规模数据表现好，处理多分类任务
      3. 对缺失数据不敏感，算法简单，常用于文本分类
    - 缺点：
      1. 要假设属性之间相互独立
      2. 要知道先验概率
      3. 由先验和数据决定后验从而分类，分类决策存在一定错误率
      4. 对输入数据对表达形式敏感
- 数据聚类：
  - KMeans算法：
    - 步骤：
      1. 在数据集中初始k个簇类中心，对应k个初始簇类
      2. 计算给定数据集中每条数据到k个簇类中心距离
      3. 按照距离最近原则，将每条数据都划分到最近簇类中
      4. 更新每个簇类中心
      5. 迭代执行步骤b～d,直至簇类中心不再改变或变化小于给定误差区间，或达到迭代次数
      6. 结束算法，输出最后簇类中心和对应簇类
    - 优点：计算复杂度，思路简单
    - 缺点：k值选择，凹集不友好，异常值敏感
  - 二分KMeans算法
    - 步骤：
      1. 初始化簇类表，使之包含所有数据
      2. 对每个簇类应用k均值聚类k=2
      3. 对误差最大对簇类继续k均值<img src="http://latex.codecogs.com/gif.latex?SSE%3D%5Csum_%7Bi%3D1%7D%5En%28y_i-%5Coverline%20y%29%5E2"/>
      4. 迭代b,c，簇类数达到k停止
    - 优点：
      1. 加速kmeans速度，减少相似度计算次数
      2. 可克服kmeans收敛局部最优
  - DBSCAN
    - 基本概念：
      1. 核心对象：r邻域内的点数量不小于minPts
      2. <img src="http://latex.codecogs.com/gif.latex?%5Cxi"/>-邻域距离阈值：设定半径r
      3. 直接密度可达：p在q的r邻域内，q是核心点，则p-q直接密度可达
      4. 密度可达：q0,q1,...,qk,其中qi - qi-1直接密度可达，q0-qk密度可达
      5. 边界点：属于某一类的非核心点
      6. 噪声点：不属于任何一类簇的点
    - 步骤：
      ```md
      * 标记所有对象为unvisted;
      * DO
      * 随机选择一个unvisited对象p
      * 标记p为visited
      * if p 为核心对象
        * 创建一个新簇C,把p添加到C
        * 令N为p的邻域中对象集合
        * For N 中每个点p'
            * If p' 是unvisited
                * 标记 p' 为visited
                * If p' 为核心对象，把其密度可达点添加到N
                * 如果 p' 还不是任何簇族成员，把p'添加到C
        * End for
        * 输出C
      * Until 标记为unvisited为噪声
      ```
    - 思路：某点与其密度可达为一类，继续啊探索
    - 参数选择：
      - 半径r:可以根据k距离设定，找到突变点
      - k距离:给定数据集p={p(i),i=0,1,...,n},计算点p(i)到集合D到子集S中所有点之间点距离，距离按照从小到大顺序排序，d(k)就被称为k-距离
      - minPts:k-距离中k值，一般取小一些，多次尝试
    - 优点：
      1. 不需指定簇个数
      2. 可以发现任何形式簇
      3. 擅长找离群点
    - 缺点：
      1. 高维数据困难
      2. 参数难以选择
  - 聚类评估
    - 紧密性（Compactness,CP）：计算每一类各点到聚类中心平均距离<img src="http://latex.codecogs.com/gif.latex?%5Coverline%20%7BCP%7D%20%3D%5Cfrac%7B1%7D%7Bk%7D%20%5Csum_%7Bk%3D1%7D%5Ek%20%5Coverline%20%7BCP%7D_k%20%5C%20%5C%20%5C%20%5Coverline%20%7BCP%7D_k%3D%5Cfrac%7B1%7D%7BC_L%7D%5Csum_%7Bx_i%5Cin%20C%7D%7Cx_i-c_i%7C"/>
    - 间隔性（Separation,SP）:计算各聚类中心两两之间平均距离<img src="http://latex.codecogs.com/gif.latex?%5Coverline%20%7BSP%7D%3D%5Cfrac%7B2%7D%7Bk%5E2-k%7D%5Csum_%7Bi%3D1%7D%5Ek%5Csum_%7Bj%3Di&plus;1%7D%5Ek%7Cw_i-w_j%7C"/>
    - 戴维森堡丁指数（Davies-Bouldin Index,DB）:计算任意两类别的类内平均距离之和除以两聚类中心距离求最大值<img src="http://latex.codecogs.com/gif.latex?DB%3D%5Cfrac%7B1%7D%7Bk%7D%5Csum_%7Bi%3D1%7D%5Ek%20%5Cmax_%7Bi%5Cne%20j%7D%28%5Cfrac%7B%5Coverline%20C_i%20&plus;%20%5Coverline%20C_j%7D%7B%7Cw_i-w_j%7C%7D%29"/>
    - 邓恩指数（Dunn Validity Index,DVI）:计算任意两个簇族元素最短距离(类间)除以任意簇类中最大距离(类内)，<img src="http://latex.codecogs.com/gif.latex?DVI%3D%5Cfrac%7B%5Cmin_%7B0%20%3C%20m%20%5Cneq%20n%20%3C%20k%7D%5C%7B%5Cmin_%7B%5Cforall%20x_i%20%5Cin%20C_m%2C%5Cforall%20x_j%20%5Cin%20C_n%7D%7Cx_i-x_j%7C%5C%7D%7D%7B%5Cmax_%7B0%20%3C%20m%20%5Cle%20k%7D%5Cmax_%7B%5Cforall%20x_i%2Cx_j%20%5Cin%20C_m%7D%7Cx_i-x_j%7C%7D"/>其中<img src="http://latex.codecogs.com/gif.latex?%7Cx_i-x_j%7C"/>表示任意两簇中元素之间距离
    - 准确性（Cluster accuracy,CA）:计算正确聚类数据数目占总数据数目比例，<img src="http://latex.codecogs.com/gif.latex?CA%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5Ek%20C_i"/>
- 关联分析--Apriori算法<br>
  找出物品之间关联度
  - 定义：
    - 项集支持度：一个项集出现次数与数据集所有事物数百分比，<img src="http://latex.codecogs.com/gif.latex?support%28A-%3EB%29%3D%5Cfrac%7BsupportCount%28A%20%5Ccup%20B%29%7D%7BN%7D"/>
    - 项集置信度：数据集中同时包含A，B数目占A比例，<img src="http://latex.codecogs.com/gif.latex?confidence%28A-%3EB%29%3D%5Cfrac%7BsupportCount%28A%20%5Ccup%20B%29%7D%7BsupportCount%28A%29%7D"/>
  - 步骤：
    1. 通过扫描数据库，累计每个项计数，收集满足最小支持度的项<img src="http://latex.codecogs.com/gif.latex?support%28A-%3EB%29%20%3D%20%5Cfrac%7BsupportCount%28A%29%7D%7BN%7D%20%3E%20%5Calpha"/>，找出单物品集合，记L1
    2. 使用L1找出共同出现2个物品项集集合L2，L2找L3...
    3. 如此下去直到不能再找到可共同出现物品项集，每找出一个Lk需一次完整数据库扫描   