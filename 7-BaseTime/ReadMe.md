### 基于上下文推荐
- 时间效应
  1. 偏好迁移：随时间对偏好发生改变
  2. 生命周期：事物热度周期
  3. 季节效应：事物流行度与季节强相关
  4. 节日选择：不同节日对用户选择产生影响
- 时间效应分析：
  1. 个人兴趣度随时间发生变化
  2. 物品流行度随时间发生变化
  3. 社会群体兴趣度随时间发生变化
- 协同过滤时间因子
  - UserCF:
    1. 用户相似度：<br>
       <img src="http://latex.codecogs.com/gif.latex?w_%7Buv%7D%3D%5Cfrac%7B%5Csum_%7Bi%20%5Cin%20N%28u%29%20%5Ccap%20N%28v%29%7D%5Cfrac%7B1%7D%7Blg%281&plus;N%28i%29%29%7Df%28%7Ct_%7Bui%7D-t_%7Bvi%7D%7C%29%7D%7B%7CN%28u%29%20%5Ccup%20N%28v%29%7C%7D"/><br>
       分母表示用户u和v产生行为并集数<br>
       <img src="http://latex.codecogs.com/gif.latex?N%28u%29%20%5Ccap%20N%28v%29"/>表示用户u和v产生行为物品交集数<br>
       <img src="http://latex.codecogs.com/gif.latex?f%28%7Ct_%7Bui%7D-t_%7Bvi%7D%7C%29%3D%5Cfrac%7B1%7D%7B1&plus;%5Calpha%20%7Ct_%7Bui%7D-t_%7Bvi%7D%7C%7D"/><br>
       <img src="http://latex.codecogs.com/gif.latex?%5Calpha"/>表示时间衰减因子<br>
        <img src="http://latex.codecogs.com/gif.latex?t_%7Bui%7D%20%5C%20t_%7Bvi%7D"/>表示用户u，v对i发生行为时间
    2. 用户u对物品i偏好：<br>
       <img src="http://latex.codecogs.com/gif.latex?r_%7Bui%7D%3Dw_%7Buv%7D%5Ctimes%20r_%7Bvi%7D%20%5Ctimes%20f%28%7Ct_0-t_vi%7C%29"/><br>
        <img src="http://latex.codecogs.com/gif.latex?f%28%7Ct_0-t_vi%7C%29%3D%5Cfrac%7B1%7D%7B1&plus;%5Cbeta%20%7Ct_0-t_%7Bvi%7D%7C%7D"/><br>
        <img src="http://latex.codecogs.com/gif.latex?t_0"/>：当前时间
  - ItemCF:
    1. 物品相似度：<br>
      <img src="http://latex.codecogs.com/gif.latex?w_%7Bij%7D%3D%5Cfrac%7B%5Csum_%7Bu%20%5Cin%20N%28i%29%20%5Ccap%20N%28j%29%7D%5Cfrac%7B1%7D%7Blg%281&plus;N%28u%29%29%7Df%28%7Ct_%7Bui%7D-t_%7Buj%7D%7C%29%7D%7B%5Csqrt%7B%7CN%28i%29%7C%7CN%28j%29%7C%7D%7D"/><br>
      <img src="http://latex.codecogs.com/gif.latex?f%28%7Ct_%7Bui%7D-t_%7Buj%7D%7C%29%3D%5Cfrac%7B1%7D%7B1&plus;%5Calpha%20%7Ct_%7Bui%7D-t_%7Buj%7D%7C%7D"/><br>
      <img src="http://latex.codecogs.com/gif.latex?N%28i%29%20%5Ccap%20N%28j%29"/>表示物品i和物品j共同产生行为对用户数<br>
      <img src="http://latex.codecogs.com/gif.latex?N%28i%29%7C%7CN%28j%29"/>表示物品i和物品j产生行为用户并集<br>
      N(u)表示用户u评分物品集合<br>
      <img src="http://latex.codecogs.com/gif.latex?t_%7Bui%7D%20%5C%20t_%7Buj%7D"/>表示用户u对i和j产生行为时间
    2. 用户u对物品j偏好：<br>
      <img src="http://latex.codecogs.com/gif.latex?r_%7Buj%7D%3Dw_%7Bij%7D%5Ctimes%20r_%7Bui%7D%20%5Ctimes%20f%28%7Ct_0-t_%7Bui%7D%7C%29"/><br>
      <img src="http://latex.codecogs.com/gif.latex?f%28%7Ct_0-t_%7Bui%7D%7C%29%3D%5Cfrac%7B1%7D%7B1&plus;%5Cbeta%20%7Ct_0-t_%7Bui%7D%7C%7D"/>
- 基于地域和热度特征推荐
  - 地域：<br>
    <img src="http://latex.codecogs.com/gif.latex?RecScore%28u%2Ci%29%3DP%28u%2Ci%29-TravelPenalty%28u%2Ci%29"/><br>
    P(u,i)：用户对物品i评分<br>
    TravelPenalty(u,i)：基本处理思路：对于u之前评分所有位置，计算距离平均值或最小值，然后对用户u当前位置i距离进行归一化
  - 热度：<br>
    最终热度：<img src="http://latex.codecogs.com/gif.latex?S%20%3D%20%5Cfrac%7BS_0&plus;S_1%7D%7BS_2%7D"/><br>
    S0:初始化热度分，不同类别不同时间都不同，可维护热点词库，与热点词匹配得到初始值<br>
    S1:与用户交互热度S1，随新闻不断被点击转发等而增加<br>
    S2:时效性热度，会随时间衰减
        
      

