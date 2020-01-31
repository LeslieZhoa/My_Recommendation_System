## My RS
*this is my own RS learning*
- ### Collaborative Filtering  --> 拿基于用户CF(协同过滤)举例:
  - 计算其他用户与目标用户的相似度sim.
    - 相似度计算方法有:(1)计算欧几里得距离;(2)计算皮尔逊相关系数;(3)计算cosine相似度
    - 拿cosine相似度推荐电影来举例  <img src="http://latex.codecogs.com/gif.latex?C%28x%2Cy%29%3D%5Cfrac%7B%5Csum%20x_i%20y_i%7D%7B%5Csqrt%7B%5Csum%7Bx_i%5E2%7D%7D%5Csqrt%7B%5Csum%7By_i%5E2%7D%7D%7D"/>   <img src="http://latex.codecogs.com/gif.latex?x_i"/>  和<img src="http://latex.codecogs.com/gif.latex?y_i"/> 分别是为相同电影评分两用户的打分
  - 计算目标用户未观看电影推荐分数
    - 找到其他用户评过分而目标用户没有看过或评分的电影item,其他用户的评分x与上述求的的sim加权求和,得到针对该目标用户的各类电影推荐分数
  - 由推荐分数推荐合适电影
  - 缺点:
    - 数据稀疏，随着数据量增大计算top k时间增加
    - 初始数据评分少会导致难以精确计算top k
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
    ```math
    A_0 = x
    
    Z_i=W_iA_{i-1}
    
    A_i=sigmoid(Z_i)
    
    y = A_n
    
    loss = cross\_entropy
    ```
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
        ```math
        For \ each \  dimension \  i \ \ \ (q_i \ z_i \ init \ 0)
        
        \sigma_i = \frac{1}{\alpha}\sqrt{q_i+g_i^2}-\sqrt{q_i} \ \ \ q_i=q_i+g_i^2 \ \ \ \Leftarrow \frac{1}{\eta^t}-\frac{1}{\eta^{t-1}}
        
        z_i = z_i+g_i-\sigma_i w_i
        
        w_i \begin{cases}0 & if |z_i^t|\leq lambda_1 \\ -(lambda_2+\frac{\beta+\sqrt{q_i}}{\alpha})^{-1}(z_i-\lambda_1 sgn(z_i)) & otherwise \end{cases}
        ```
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
    
    - 特点：从原始数据中同时学习到了低维与高维特征；
不再需要特征工程
    
        
      

