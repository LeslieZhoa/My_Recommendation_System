### 基于用户行为特征推荐<br>
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
    3. 计算推荐结果<img src="http://latex.codecogs.com/gif.latex?p%28u%2Ci%29%3D%5Csum_%7Bj%20%5Cin%20S%28i%2Ck%29%20%5Ccap%20N%28u%29%7D%20W_%7Bij%7Dr_%7Buj%7D"/><br>
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