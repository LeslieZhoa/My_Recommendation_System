## My RS
   *this is my own RS learning*
- ### 基于用户行为特征推荐<br>
- 基于内容对推荐算法
  - 步骤：
      1. 构造Item特征
      2. 计算Item之间对相似度<img src="http://latex.codecogs.com/gif.latex?cos%28U%2CI%29%3D%5Cfrac%7B%5Csum%20U_a*I_a%7D%7B%5Csqrt%7B%5Csum%20U_a%5E2%7D%5Csqrt%7B%5Csum%20I_a%5E2%7D%7D"/>其中<img src="http://latex.codecogs.com/gif.latex?U_a"/>:用户对电影类型a的偏好，<img src="http://latex.codecogs.com/gif.latex?I_a"/>电影是否属于类型a
      3. 评判用户是否喜欢某个item
  - 优点：
      1. 不需要其他用户数据
      2. 没有冷开始问题和稀疏问题
  - 缺点：
      1. 要求内容能容易抽成有意义特征
      2. 用户品味必须可用内容特征表示 
- UserCF<br>
    先找到相似用户，再找到他们喜欢物品
  - 步骤：
      1. 计算其他用户与目标用户的相似度sim.
         相似度计算方法有:(1)计算欧几里得距离;(2)计算皮尔逊相关系数;(3)计算cosine相似度
      2. 拿cosine相似度推荐电影来举例<img src="http://latex.codecogs.com/gif.latex?C%28x%2Cy%29%3D%5Cfrac%7B%5Csum%20x_i%20y_i%7D%7B%5Csqrt%7B%5Csum%7Bx_i%5E2%7D%7D%5Csqrt%7B%5Csum%7By_i%5E2%7D%7D%7D"/> <img src="http://latex.codecogs.com/gif.latex?x_i"/>和<img src="http://latex.codecogs.com/gif.latex?y_i"/>分别是为相同电影评分两用户的打分
      3. 计算目标用户未观看电影推荐分数
      4. 找到其他用户评过分而目标用户没有看过或评分的电影item,其他用户的评分x与上述求的的sim加权求和,得到针对该目标用户的各类电影推荐分数
      5. 由推荐分数推荐合适电影
  - 优化：
      1. 很多用户没有交集，没有计算必要
         1. 建立物品到用户倒排表T,表示该物品被哪些用户发生过行为
         2. 根据倒排表T，建立用户相似度矩阵U
      2. 惩罚热门物品<img src="http://latex.codecogs.com/gif.latex?W_%7Buv%7D%3D%5Cfrac%7B%5Csum_%7Bi%20%5Cin%20N%28u%29%20%5Ccap%20N%28v%29%7D%5Cfrac%7B1%7D%7Blg%281&plus;%7CN%28i%29%7C%29%7D%7D%7B%5Csqrt%7B%7CN%28u%29%7C%7CN%28v%29%7C%7D%7D"/>
      3. 最终概率计算公式<img src="http://latex.codecogs.com/gif.latex?p%28u%2Ci%29%3D%5Csum_%7Bv%20%5Cin%20S%28u%2Ck%29%20%5Ccap%20N%28i%29%7D%20W_%7Buv%7Dr_%7Bvi%7D"/><br>
      <img src="http://latex.codecogs.com/gif.latex?p%28u%2Ci%29"/>表示用户u对物品i对感兴趣程度;<br>
      <img src="http://latex.codecogs.com/gif.latex?S%28u%2Ck%29"/>表示和用户u兴趣最接近的k个用户;<br>
      <img src="http://latex.codecogs.com/gif.latex?N%28i%29"/>表示对物品i有过行为的用户集合;<br>
      <img src="http://latex.codecogs.com/gif.latex?W_%7Buv%7D"/>表示用户u和用户v兴趣相似度<br>
      <img src="http://latex.codecogs.com/gif.latex?r_%7Bui%7D"/>表示用户u对物品i对兴趣
  - 特点：
      1. 当物品数量远超用户使用
      2. 更注重热门物品推荐
- ItemCF<br>
    先找到用户喜欢物品，在找到喜欢物品对相似物品
  - 步骤：
      1. 计算物品之间相似度<img src="http://latex.codecogs.com/gif.latex?W_%7Bij%7D%3D%5Cfrac%7B%7CN%28i%29%20%5Ccap%20N%28j%29%7C%7D%7B%7CN%28i%29%7C%7D"/>其中<img src="http://latex.codecogs.com/gif.latex?%7CN%28i%29%7C">表示喜欢物品i对用户数
      2. 惩罚热门商品<img src="http://latex.codecogs.com/gif.latex?W_%7Bij%7D%3D%5Cfrac%7B%7CN%28i%29%20%5Ccap%20N%28j%29%7C%7D%7B%5Csqrt%7B%7CN%28i%29%7C%7CN%28j%29%7C%7D%7D"/>
      3. 计算推荐结果<img src="http://latex.codecogs.com/gif.latex?p%28u%2Ci%29%3D%5Csum_%7Bv%20%5Cin%20S%28i%2Ck%29%20%5Ccap%20N%28u%29%7D%20W_%7Bij%7Dr_%7Bui%7D%24"/><br>
      <img src="http://latex.codecogs.com/gif.latex?p%28u%2Ci%29"/>表示用户u对物品i对感兴趣程度;<br>
      <img src="http://latex.codecogs.com/gif.latex?S%28i%2Ck%29"/>表示和物品i最相似的k个物品;<br>
      <img src="http://latex.codecogs.com/gif.latex?W_%7Bij%7D"/>表示物品i和物品j相似度<br>
      <img src="http://latex.codecogs.com/gif.latex?r_%7Buj%7D"/>表示用户u对物品j对兴趣
  - 特点：
      1. 当用户数量远超物品使用
      2. 具有很好新颖性，精确率略低
- CF缺点:
  - 数据稀疏，随着数据量增大计算top k时间增加
  - 初始数据评分少会导致难以精确计算top k
- 基于隐语义模型的推荐算法LFM
  - 算法：<img src="http://latex.codecogs.com/gif.latex?R%28u%2Ci%29%3D%5Csum_%7Bk%3D1%7D%5Ek%20P_%7Bu%2Ck%7DQ_%7Bi%2Ck%7D"/><br>
             <img src="http://latex.codecogs.com/gif.latex?P_%7Bu%2Ck%7D"/>:用户u兴趣和第k个隐类关系<br>
             <img src="http://latex.codecogs.com/gif.latex?Q_%7Bi%2Ck%7D"/>:第k个隐类和物品i的关系<br>
             k：隐类的数量<br>
             R：用户对物品对兴趣度
  - 目的：自定义k值，通过大量数据，求出P，Q矩阵
  - 损失函数：<img src="http://latex.codecogs.com/gif.latex?loss%3D%5Csum_%7B%28u%2Ci%29%20%5Cin%20S%7D%28R_%7Bui%7D-%5Chat%20R_%7Bui%7D%29%5E2"/>
  - 优点：
      1. 基于用户行为自动聚类，可反映用户对物品分类意见
      2. 隐语义模型能动态获取用户兴趣类别和程度
      3. 隐语义模型能计算出物品在各个类别中权值
  - 缺点：很难实时，计算比较耗时
- ### Word2Vec
- 一个衍生品，预测单词上下文网络的权重信息，可以表示单词的编码信息
- 常见方法：
  - CBOW:根据上下文单词预测中间单词
  - skip-gram:根据中间词，预测上下文信息
  - GloVe:目的是找出两个词i,j一起出现的频率最大
- 拿skip-gram举例算法流程
  - 词字典搭建
    - 由一篇文章之类的文本得到词字典[单词:index],index从零计数，出现频率大的单词index越小。设置阈值，频率过小记为unkonw
  - 获取训练数据
    - 假设上下文的window_size为2，对应划窗的五个单词为['a','b','c','d','e'],对应index为[0,1,2,3,4],所获取的[input,label]对选取的单词可能是['c','a'],['c','d'],对应值为[2,0],[2,3],其中input为中心词，label为随机选取的划窗内的上下文词
  - 搭建网络
    - 主要是embeeding的变量，shape为[vocabulary_size,embed_size]
  - 损失函数 nce loss 为例
    - 1.随机选取与label的index不相同的index即负样本的index计作neg，其中高频词的几率更大。         选取方式<img src="http://latex.codecogs.com/gif.latex?p(k)=(log(k+2)-log(k+1))/log(vocabulary_size+1)"/> k越小即频数越大，选中概率越大
    - 2.将label与neg的index concat在一起得到all_index，选取nce_weights与nce_biases的all_index部分与embed做矩阵运算得到logits；对于每个batch的 all_index的第一维为true,其他维为false，所以对应的label
            应为[1,0,0,...]
    - 3.将label和logits做交叉熵得到最终loss
  - 最小化loss得到embedding矩阵
    - embedding矩阵的第index行即为该单词的词向量，可根据词向量计算cosine距离来判定相似词语
- ### 逻辑回归LR
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
- ### Online Optimization
    *目的产生稀疏权重，每k步进行一次截断*
- 梯度下降方法
  - 简单截断法：
    - 表示方式：<img src="http://latex.codecogs.com/gif.latex?w%5E%7Bt&plus;1%7D%3DT_0%28w%5Et-%5Ceta%5EtG%5Et%2C%5Ctheta%29"/><br>
    &#8194; &#8194; &#8194; &#8194; &#8194; &#8194; <img src="http://latex.codecogs.com/gif.latex?T_0%28v_i%2C%5Ctheta%29%5Cbegin%7Bcases%7D0%20%26%20if%20%7Cv_i%7C%5Cleq%5Ctheta%20%5C%5Cv_i%20%26%20otherwise%20%5Cend%7Bcases%7D"/>
  - 阶段梯度法TG
    - 表示方法：<img src="http://latex.codecogs.com/gif.latex?w%5E%7Bt&plus;1%7D%3DT_1%28w%5Et-%5Ceta%5EtG%5Et%2C%5Ceta%5Et%5Clambda%5Et%2C%5Ctheta%29"/><br>
    &#8194; &#8194; &#8194; &#8194; &#8194;&#8194; <img src="http://latex.codecogs.com/gif.latex?T_1%28v_i%2C%5Calpha%2C%5Ctheta%29%5Cbegin%7Bcases%7Dmax%280%2Cv_i-%5Calpha%20%26%20if%20v_i%5Cin%20%5B0%2C%5Ctheta%5D%20%5C%5Cmin%280%2Cv_i&plus;%5Calpha%20%26%20if%20v_i%5Cin%20%5B-%5Ctheta%2C0%5D%5C%5C%20v_i%20%26%20otherwise%20%5Cend%7Bcases%7D"/>
    - 与T0差别：相较于T0的一个参数，多了一个参数进行截断，过度相对平缓
  - 前向后向切分FOBOS
    - 表示方法：L1-FOBOS举例<img src="http://latex.codecogs.com/gif.latex?w_i%5E%7Bt&plus;1%7D%5Cbegin%7Bcases%7D0%20%26%20if%20%7Cw_i%5Et-%5Ceta%5Etg_i%5Et%7C%20%5Cleq%20%5Ceta%5E%7Bt&plus;%5Cfrac%7B1%7D%7B2%7D%7D%20%5Clambda%20%5C%5C%20%28w_i%5Et-%5Ceta%5Et%20g_i%5Et-%5Ceta%5E%7Bt&plus;%5Cfrac%7B1%7D%7B2%7D%7D%5Clambda%20sgn%28w_i%5Et-%5Ceta%5Etg_i%5Et%29%20%26%20otherwise%20%5Cend%7Bcases%7D"/> 
    - 特点：相较于T0,T1和某值进行比较截断，该方法稀疏条件--><img src="http://latex.codecogs.com/gif.latex?%7Cw_i%5Et-%5Ceta%5Etg_i%5Et%7C%20%5Cleq%20%5Ceta%5E%7Bt&plus;%5Cfrac%7B1%7D%7B2%7D%7D%20%5Clambda"/>，当一条样本梯度不足以令维度上当权重值发生足够大变化<img src="http://latex.codecogs.com/gif.latex?%5Ceta%5E%7Bt&plus;%5Cfrac%7B1%7D%7B2%7D%7D%20%5Clambda"/>，则在本次更新不重要，令权重·为0
  - 累计梯度方法
    - 正则对偶平均RDA
      - 表示方式：L1-RDA举例<img src="http://latex.codecogs.com/gif.latex?w_i%5E%7Bt&plus;1%7D%5Cbegin%7Bcases%7D0%20%26%20if%20%7C%5Cbar%7Bg%7D_i%5Et%7C%5Cleq%20%5Clambda%20%5C%5C%20-%5Cfrac%7Bt%7D%7B%5Cgamma%7D%28%5Cbar%7Bg%7D_i%5Et%20-%20%5Clambda%20sgn%28%5Cbar%7Bg%7D_i%5Et%29%29%20%26%20otherwise%20%5Cend%7Bcases%7D"/>
      - 特点：
        - L1-FOBOS截断条件<img src="http://latex.codecogs.com/gif.latex?%7Cw_i%5Et-%5Ceta%5Etg%5Et_i%7C%5Cleq%20%5Ceta%5E%7Bt&plus;%5Cfrac%7B1%7D%7B2%7D%7D%5Clambda"/>其中<img src="http://latex.codecogs.com/gif.latex?%5Ceta"/>正比于<img src="http://latex.codecogs.com/gif.latex?%5Cfrac%7B1%7D%7B%5Csqrt%7Bt%7D%7D"/>，所以随着t增加，阈值绘减小，是基于梯度下降有较高精度；
        - L1-RDA截断条件是常数，更容易产生稀疏，判定条件是平均累加梯度，可以避免训练不足导致截断问题，通过调节<img src="http://latex.codecogs.com/gif.latex?%5Clambda"/>一个参数，很容易在精度和稀疏性上权衡。但更新权重也是累积权重平均，相比于梯度下降精度会有所下降。
    - FTRL<br>
      *结合FOBOS和RDA两家之长*
      - 表达方式：(以下算式中<img src="http://latex.codecogs.com/gif.latex?G%5E%7B1%3At%7D%3D%5Csum_%7Bs%3D1%7D%5EtG%5Es"/> &#8194; <img src="http://latex.codecogs.com/gif.latex?%5Csigma%5Es%3D%5Cfrac%7B1%7D%7B%5Ceta%5Es%7D-%5Cfrac%7B1%7D%7B%5Ceta%5E%7Bs-1%7D%7D"/> &#8194; <img src="http://latex.codecogs.com/gif.latex?%5Csigma%5E%7B1%3At%7D%3D%5Cfrac%7B1%7D%7B%5Ceta%5Et%7D"/> )
        - L1-FOBOS --> <img src="http://latex.codecogs.com/gif.latex?W%5E%7Bt&plus;1%7D%3Darg%5C%2C%20%5Cmin_%7BW%7D%20%5C%7B%20G%5EtW&plus;%5Clambda%7C%7CW%7C%7C_1&plus;%5Cfrac%7B1%7D%7B2%7D%5Csigma%5E%7B1%3At%7D%7C%7CW-W%5Et%7C%7C_2%5E2%5C%7D"/>
        - L1-RDA --> <img src="http://latex.codecogs.com/gif.latex?W%5E%7Bt&plus;1%7D%3Darg%5C%2C%20%5Cmin_%7BW%7D%20%5C%7B%20G%5E%7B1%3At%7DW&plus;%5Clambda%7C%7CW%7C%7C_1&plus;%5Cfrac%7B1%7D%7B2%7D%5Csigma%5E%7B1%3At%7D%7C%7CW-0%7C%7C_2%5E2%5C%7D"/>
        - 区别：梯度和梯度累加；不能离W或0太远
        - 综合两种距离，不能离W太远也不能离0太远,<img src="http://latex.codecogs.com/gif.latex?W%5E%7Bt&plus;1%7D%3Darg%5C%2C%20%5Cmin_%7BW%7D%20%5C%7B%20G%5E%7B1%3At%7DW&plus;%5Clambda_1%7C%7CW%7C%7C_1&plus;%5Clambda_2%20%7C%7CW%7C%7C_2%5E2&plus;%5Cfrac%7B1%7D%7B2%7D%5Csum_%7Bs%3D1%7D%5Et%5Csigma%5E%7Bs%7D%7C%7CW-W%5Es%7C%7C_2%5E2%5C%7D"/>
      - 算法步骤：
        <div align=center><img src="http://latex.codecogs.com/gif.latex?For%20%5C%20each%20%5C%20dimension%20%5C%20i%20%5C%20%5C%20%5C%20%28q_i%20%5C%20z_i%20%5C%20init%20%5C%200%29%20%5C%5C%20%5Csigma_i%20%3D%20%5Cfrac%7B1%7D%7B%5Calpha%7D%5Csqrt%7Bq_i&plus;g_i%5E2%7D-%5Csqrt%7Bq_i%7D%20%5C%20%5C%20%5C%20q_i%3Dq_i&plus;g_i%5E2%20%5C%20%5C%20%5C%20%5CLeftarrow%20%5Cfrac%7B1%7D%7B%5Ceta%5Et%7D-%5Cfrac%7B1%7D%7B%5Ceta%5E%7Bt-1%7D%7D%20%5C%5C%20z_i%20%3D%20z_i&plus;g_i-%5Csigma_i%20w_i%20%5C%5C%20w_i%20%5Cbegin%7Bcases%7D0%20%26%20if%20%7Cz_i%5Et%7C%5Cleq%20lambda_1%20%5C%5C%20-%28lambda_2&plus;%5Cfrac%7B%5Cbeta&plus;%5Csqrt%7Bq_i%7D%7D%7B%5Calpha%7D%29%5E%7B-1%7D%28z_i-%5Clambda_1%20sgn%28z_i%29%29%20%26%20otherwise%20%5Cend%7Bcases%7D"/></div>
- ### 因式分解机FM<br>
  *多用于ctr问题*
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
    
  - 特点：从原始数据中同时学习到了低维与高维特征；不再需要特征工程
- ### 数据挖掘
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
        
      

