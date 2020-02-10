import pandas as pd  
import math 

class RecBasedTag:
    def __init__(self):
        # 用户听过艺术家次数
        self.user_rate_file = 'data/user_artists.dat'
        # 用户打标签信息
        self.user_tag_file = 'data/user_taggedartists.dat'
        # 获取所有艺术家ID
        self.artistAll = list(
            pd.read_table('data/artists.dat',delimiter="\t")["id"].values
        ) 

        # 用户对艺术家评分
        self.userRateDict = self.getUserRate()
        # 艺术家与标签相关度，每个用户打标的标签，每个标签被所有用户打标次数
        self.artistsTagsDict, self.userTagDict, self.tagUserDict = self.getTags()
        # 用户对每个标签喜好度
        self.userTagPre = self.getUserTagPre()
    def getUserRate(self):
        # 获取用户对艺术家评分信息
        userRateDict = dict()
        with open(self.user_rate_file,'r',encoding='utf-8') as fr:
            for line in fr.readlines():
                if not line.startswith('userID'):
                    userID, artistID, weight = line.split('\t')
                    userRateDict.setdefault(int(userID),{})

                    # 对听歌次数缩放，避免结果太大
                    userRateDict[int(userID)][int(artistID)] = float(weight) / 10000
        return userRateDict 

    def getTags(self):

        # 艺术家对应标签基因，有相关度为1，否则为0
        artistsTagsDict = dict()
        # 获取每个用户打标的标签
        tagUserDict = dict()
        # 获取每个标签被所有用户打标次数
        userTagDict = dict()
        for line in open(self.user_tag_file,"r",encoding="utf-8"):
            if not line.startswith('userID'):
                userID, artistID, tagID = line.strip().split('\t')[:3]
                # 赋值艺术家对应标签基因
                artistsTagsDict.setdefault(int(artistID),{})
                artistsTagsDict[int(artistID)][int(tagID)] = 1
                # 统计每个标签被打标次数
                if int(tagID) in tagUserDict.keys():
                    tagUserDict[int(tagID)] += 1
                else:
                    tagUserDict[int(tagID)] = 1
                # 统计每个用户对每个标签打标次数
                userTagDict.setdefault(int(userID),{})
                if int(tagID) in userTagDict[int(userID)].keys():
                    userTagDict[int(userID)][int(tagID)] += 1
                else:
                    userTagDict[int(userID)][int(tagID)] = 1
        return artistsTagsDict,userTagDict,tagUserDict 

    def getUserTagPre(self):
        # 获取用户对标签对最终兴趣度
        userTagPre = dict()
        # 用户打标次数
        userTagCount = dict()

        with open(self.user_tag_file,"r",encoding='utf-8') as fr:
            lines = fr.readlines()
            # 用户打标总条数
            Num = len(lines)
            for line in lines:
                if not line.startswith('userID'):
                    userID, artistID, tagID = line.split('\t')[:3]

                    userTagPre.setdefault(int(userID),{})
                    userTagCount.setdefault(int(userID),{})

                    # 用户为艺术家打分
                    rate_ui = (
                        self.userRateDict[int(userID)][int(artistID)]
                        if int(artistID) in self.userRateDict[int(userID)].keys()
                        else 0
                    )

                    if int(tagID) not in userTagPre[int(userID)].keys():
                        userTagPre[int(userID)][int(tagID)] = (
                            rate_ui * self.artistsTagsDict[int(artistID)][int(tagID)]
                        )
                        userTagCount[int(userID)][int(tagID)] = 1
                    else:
                        userTagPre[int(userID)][int(tagID)] += (
                            rate_ui * self.artistsTagsDict[int(artistID)][int(tagID)]
                        )
                        userTagCount[int(userID)][int(tagID)] += 1

            for userID in userTagPre.keys():
                for tagID in userTagPre[userID].keys():
                    tf_ut = self.userTagDict[int(userID)][int(tagID)] / sum(
                        self.userTagDict[int(userID)].values()
                    )
                    idf_ut = math.log(Num * 1.0 / (self.tagUserDict[int(tagID)] + 1))

                    userTagPre[userID][tagID] = (
                        userTagPre[userID][tagID]/userTagCount[userID][tagID] * tf_ut * idf_ut
                    )
            return userTagPre

        
    def recommendForUser(self,user,K,flag=True):
        # 根据用户对标签喜好得到相应艺术家推荐
        
        # 用户艺术家喜好度
        userArtistPreDict = dict()
        for artist in self.artistAll:
            if int(artist) in self.artistsTagsDict.keys():
                #计算用户对标签喜好度
                for tag in self.userTagPre[int(user)].keys():
                    rate_ui = self.userTagPre[int(user)][int(tag)]
                    # 艺术家对标签相关性
                    rel_it = (
                        0
                        if tag not in self.artistsTagsDict[int(artist)].keys()
                        else self.artistsTagsDict[int(artist)][tag]
                    )
                    if artist in userArtistPreDict.keys():
                        userArtistPreDict[int(artist)] += rate_ui * rel_it
                    else:
                        userArtistPreDict[int(artist)] = rate_ui * rel_it
        newUserArtistPreDict = dict()
        if flag:
            # 过滤掉已听过艺术家
            for artist in userArtistPreDict.keys():
                if artist not in self.userRateDict[int(user)].keys():
                    newUserArtistPreDict[artist] = userArtistPreDict[int(artist)]
            return sorted(
                newUserArtistPreDict.items(),key=lambda k: k[1],reverse=True
            )[:K]

        else:
            return sorted(userArtistPreDict.items(),key=lambda k: k[1],reverse=True)[:K]

    def evaluate(self,user):
        K = math.ceil(len(self.userRateDict[int(user)])/2)
        recResult = self.recommendForUser(user,K=K,flag=False)
        count = 0
        for (artist,_) in recResult:
            if artist in self.userRateDict[int(user)]:
                count += 1
        return count * 1.0 / K

if __name__ == "__main__":
    rbt = RecBasedTag()
    print('开始推荐： ',rbt.recommendForUser("2",K=3))
    print('验证： ',rbt.evaluate("2"))


