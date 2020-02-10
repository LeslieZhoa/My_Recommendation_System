import jieba 
import math 

class TF_IDF:
    def __init__(self,file,stop_file):
        self.file = file
        self.stop_file = stop_file 
        self.stop_words = self.getStopWords()

    def getStopWords(self):
        # 获取标点等无关词语
        swlist = list()
        for line in open(self.stop_file,'r',encoding='utf-8').readlines():
            swlist.append(line.strip())
        print('加载无关词语集完成...')
        return swlist

    def loadData(self):
        # 加载商品对应短语，并用jieba分词，除去无用词语
        dMap = dict()
        for line in open(self.file,'r',encoding='utf-8').readlines():
            id, title = line.strip().split('\t')
            dMap.setdefault(id,[])
            for word in list(jieba.cut(str(title).replace(' ',''),cut_all=False)):
                if word not in self.stop_words:
                    dMap[id].append(word)
        print('加载完成商品短语...')
        return dMap

    def getFreqWord(self,words):
        # 获取words中短语频次
        freqWord = dict()
        for word in words:
            freqWord.setdefault(word,0)
            freqWord[word] += 1
        return freqWord 

    def getCountWordInFile(self,word,dMap):
        # 统计word出现在dMap的频次
        count = 0
        for key in dMap.keys():
            if word in dMap[key]:
                count += 1
        return count
     
    def getTFIDF(self,words,dMap):
        # 获取words中word的tfidf
        outDic = dict()
        freqWord = self.getFreqWord(words)
        for word in words:
            tf = freqWord[word] * 1.0 / len(words)

            idf = math.log(len(dMap) / (self.getCountWordInFile(word,dMap) + 1.0))
            tfidf = tf * idf 
            outDic[word] = tfidf 
        # 按tfidf值排序
        orderDic = sorted(outDic.items(), key=lambda  x: x[1], reverse=True)
        return orderDic

if __name__ == "__main__":
    # id对应短语文件
    file = 'data/id_title.txt'
    # 无关短语文件
    stop_file = 'data/stop_words.txt'
    tfidf = TF_IDF(file,stop_file)

    dMap = tfidf.loadData()
    for id in dMap.keys():
        tfIdfDic = tfidf.getTFIDF(dMap[id],dMap)
        item_list = list(filter(lambda x:x[1]>0.2,tfIdfDic))
        item_list = list(map(lambda  x:x[0],item_list))
        print("id:{}|feature:{}".format(id," ".join(item_list)))
        
