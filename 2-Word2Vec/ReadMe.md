Word2Vec
*算法代码参考https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/word2vec*
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