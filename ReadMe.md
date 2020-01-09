## My RS
*this is my own RS learning*
- Collaborative Filtering  --> 拿基于用户CF(协同过滤)举例:
  - 计算其他用户与目标用户的相似度sim.
    - 相似度计算方法有:(1)计算欧几里得距离;(2)计算皮尔逊相关系数;(3)计算cosine相似度
    - 拿cosine相似度推荐电影来举例  <img src="http://latex.codecogs.com/gif.latex?C%28x%2Cy%29%3D%5Cfrac%7B%5Csum%20x_i%20y_i%7D%7B%5Csqrt%7B%5Csum%7Bx_i%5E2%7D%7D%5Csqrt%7B%5Csum%7By_i%5E2%7D%7D%7D"/>   <img src="http://latex.codecogs.com/gif.latex?x_i"/>  和<img src="http://latex.codecogs.com/gif.latex?y_i"/> 分别是为相同电影评分两用户的打分
  - 计算目标用户未观看电影推荐分数
    - 找到其他用户评过分而目标用户没有看过或评分的电影item,其他用户的评分x与上述求的的sim加权求和,得到针对该目标用户的各类电影推荐分数
  - 由推荐分数推荐合适电影
  - 缺点:
    - 数据稀疏，随着数据量增大计算top k时间增加
    - 初始数据评分少会导致难以精确计算top k