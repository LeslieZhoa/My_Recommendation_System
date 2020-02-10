### 基于标签推荐
- 关键词：能反映文本语料主题词语或短语
  - 类别：
    1. 标识对象的类别
    2. 标识对象的创建者或所有者
    3. 标识对象品质和特征
    4. 用户参考用到的标签
    5. 分类提炼用的标签
    6. 用于任务组织的标签
- TF-IDF算法：
  - 概念：某个词或短语在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词语或短语具有很好的类别区分能力用来分类
  - TF(term frequency)：词条t在文档Di中出现的频率<br>
    <img src="http://latex.codecogs.com/gif.latex?TF_%7Bt%2CD_i%7D%3D%5Cfrac%7Bcount%28t%29%7D%7B%7CD_i%7C%7D"/><br>
    分子表示词条t出现次数<br>
    分母表示文档Di中所有词条个数
  - IDF(inverse doucument frequency)：表示词条t在整个语料库中的区分能力<br>
    <img src="http://latex.codecogs.com/gif.latex?IDF_%7Bt%7D%3Dlg%5Cfrac%7BN%7D%7B1&plus;%5Csum_%7Bi%3D1%7D%5EN%20I%28t%2Ct%2CD_i%29%7D"/><br>
    N:所有文档总数<br>
    <img src="http://latex.codecogs.com/gif.latex?I%28t%2CD_i%29"/>：文档Di是否包含词条t，若包含则为1，不包含为0
  - TF-IDF：<img src="http://latex.codecogs.com/gif.latex?TF-IDF_%7Bt%2CD_i%7D%3DTF_%7Bt%2CD_i%7D%5Ctimes%20IDF_t"/>
- 基于标签推荐系统：
  通过用户与商品对喜爱度推算出用户对标签的喜好程度，进而根据商品与标签关联度，为用户推荐商品
  - 对应用户对标签对喜好程度计算公式：<br>
    <img src="http://latex.codecogs.com/gif.latex?rate%28u%2Ct%29%20%3D%20%5Cfrac%7B%5Csum_%7Bi%20%5Cin%20I_u%7Drate%28u%2Ci%29%20%5Ctimes%20rel%28i%2Ct%29%7D%7B%5Csum_%7Bi%20%5Cin%20I_u%7Drel%28i%2Ct%29%7D"/><br>
    <img src="http://latex.codecogs.com/gif.latex?rate%28u%2Ci%29"/>：用户u对艺术家i的评分<br>
    <img src="http://latex.codecogs.com/gif.latex?rel%28i%2Ct%29"/>：艺术家i与标签t的相关度
  - 为减小评分行为较少引起预测误差，引入平滑因子：<br>
    <img src="http://latex.codecogs.com/gif.latex?rate%28u%2Ct%29%3D%5Cfrac%7B%5Csum_%7Bi%20%5Cin%20I_u%7Drate%28u%2Ci%29%20%5Ctimes%20rel%28i%2Ct%29&plus;%5Coverline%20r_u%20%5Ctimes%20K%7D%7B%5Csum_%7Bi%20%5Cin%20I_u%7Drel%28i%2Ct%29&plus;K%7D"/><br>
      K：平滑因子<br>
    <img src="http://latex.codecogs.com/gif.latex?%5Coverline%20r_u"/>：用户u所有评分平均值
  - 用户对标签对依赖度：<br>
    <img src="http://latex.codecogs.com/gif.latex?TF%28u%2Ct%29%3D%5Cfrac%7Bn%28u%2Ct%29%7D%7B%5Csum_%7Bt_i%20%5Cin%20T%7Dn%28u%2Ct_i%29%7D"/><br>
    <img src="http://latex.codecogs.com/gif.latex?n%28u%2Ct_i%29"/>：用户u使用标签ti标记的次数<br>
    分母表示用户u使用所有标签标记次数和<br>
    TF(u,t)：用户u使用标签t标记的频率
  - 标签对于用户的热门程度：<br>
    社会化标签使用网站存在“马太效应”，即热门越热门，冷门越冷门<br>
    采用对一个标签，使用他的用户少，某用户经常采用，则亲密度高：<br>
    <img src="http://latex.codecogs.com/gif.latex?IDF_%7Bu%2Ct%7D%3Dlg%5Cfrac%7B%5Csum_%7Bu_i%20%5Cin%20U%7D%5Csum_%7Bt_j%20%5Cin%20T%7Dn%28u_i%2Ct_j%29%7D%7B%5Csum_%7Bu_i%20%5Cin%20U%7Dn%28u_i%2Ct%29&plus;1%7D"/><br>
    分子表示用户对所有标签标记计数之和<br>
    分母表示用户对标签t标记计数和<br>
  - 用户u对标签t的兴趣：<br>
    <img src="http://latex.codecogs.com/gif.latex?Pre%28u%2Ct%29%3Drate%28u%2Ct%29%20%5Ctimes%20TF-IDF%28u%2Ct%29"/><br>
    <img src="http://latex.codecogs.com/gif.latex?TF-IDF%28u%2Ct%29%3DTF%28u%2Ct%29%20%5Ctimes%20IDF%28u%2Ct%29"/><br>
  - 用户对商品喜好矩阵：<br>
    <img src="http://latex.codecogs.com/gif.latex?T%28u%2Ci%29%3DT_u%20%5Ctimes%20T_i%5ET"/><br>
    <img src="http://latex.codecogs.com/gif.latex?T_u"/>：用户u对所有标签兴趣度矩阵<br>
    <img src="http://latex.codecogs.com/gif.latex?T_i"/>：所有商品标签基因矩阵<br>
    <img src="http://latex.codecogs.com/gif.latex?T%28u%2Ci%29"/>：用户u对所有商品喜好程度
 
        
      

