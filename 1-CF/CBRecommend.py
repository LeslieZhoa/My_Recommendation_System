import json 
import pandas as pd 
import numpy as np 
import math 
import random 
import os 

class DataProcessing:
    def __init__(self):
        pass 

    def get_movie(self,file='data/movies.csv'):
        # 获取电影id与电影名字映射关系
        self.movie_dict = {}
        with open(file,"r") as fr:
            for line in fr.readlines():
                if not line.startswith("MovieID"):
                        (indx,name) = line.split(',')[:2]
                        self.movie_dict[str(indx)] = name
        return self.movie_dict


    def prepare_item_profile(self,save_path,file='data/movies.csv'):
        if not os.path.exists(save_path):
            # 获取item即电影类型的特征信息矩阵
            items = pd.read_csv(file)
            # 所有的电影id
            item_ids = set(items["MovieID"].values)
            self.item_dict = {}
            genres_all = list()

            for item in item_ids:
                # 电影item属于的所有类型
                genres = items[items["MovieID"]==item]["Genres"].values[0].split("|")
                self.item_dict.setdefault(item,[]).extend(genres)
                genres_all.extend(genres)
            self.genres_all = set(genres_all)

            #将每个电影特征做成one-hot存入self.item_matrix
            self.item_matrix = {}
            for item in self.item_dict.keys():
                self.item_matrix[str(item)] = [0] * len(self.genres_all)
                for genre in self.item_dict[item]:
                    index = list(self.genres_all).index(genre)
                    self.item_matrix[str(item)][index] = 1
            json.dump(self.item_matrix,open(save_path,'w'))
            print("item 信息计算完成，保存路径为：{}"
              .format(save_path))
        else:
            print('load item data ...')
            self.item_matrix = json.load(open(save_path,"r"))
    
        return self.item_matrix

    def prepare_user_profile(self,save_path,file='data/ratings.csv'):
        #计算用户偏好矩阵
        if not os.path.exists(save_path):
            users = pd.read_csv(file)
            user_ids = set(users['UserID'].values)

            users_rating_dict = {}
            for user in user_ids:
                users_rating_dict.setdefault(str(user),{})
            
            # [用户][电影]=评分
            with open(file,"r") as fr:
                for line in fr.readlines():
                    if not line.startswith("UserID"):
                        (user,item,rate) = line.split(',')[:3]
                        users_rating_dict[user][item] = int(rate)

            self.user_matrix = {}

            for user in users_rating_dict.keys():
                # 用户user所有打分电影值
                score_list = users_rating_dict[user].values()

                #平均分
                avg = sum(score_list)/len(score_list)
                self.user_matrix[user] = []

                for genre in self.genres_all:
                    score_all = 0.0
                    score_len = 0
                    for item in users_rating_dict[user].keys():
                        # 判断类型是否在评分电影中
                        if genre in self.item_dict[int(item)]:
                            score_all += (users_rating_dict[user][item]-avg)
                            score_len += 1
                    # 没有评分的类型默认0
                    if score_len == 0:
                        self.user_matrix[user].append(0.0)
                    else:
                        self.user_matrix[user].append(score_all / score_len)

            json.dump(self.user_matrix,open(save_path,'w'))
            print("user 信息计算完成，保存路径为：{}"
              .format(save_path))

        else:
            print('load user data ...')
            self.user_matrix = json.load(open(save_path,'r'))
        return self.user_matrix


class CBRecommend:
    def __init__(self,K,item_path,user_path):
        self.K = K 
        process = DataProcessing()
        self.item_profile = process.prepare_item_profile(item_path)
        self.user_profile = process.prepare_user_profile(user_path)
        self.movie_dict = process.get_movie()
    def get_none_score_item(self,user):
        # 获取用户未评分电影
        items = pd.read_csv('data/movies.csv')['MovieID'].values
        data = pd.read_csv('data/ratings.csv')
        have_score_items = data[data['UserID']==user]['MovieID'].values
        none_score_items = set(items) - set(have_score_items)
        return none_score_items

    def cosUI(self,user,item):
        # 获取用户对电影喜好程度
        Uia = sum(np.array(self.user_profile[str(user)])*
                np.array(self.item_profile[str(item)]))

        Ua = math.sqrt(sum([math.pow(one,2) for one in self.user_profile[str(user)]]))
        Ia = math.sqrt(sum([math.pow(one,2) for one in self.item_profile[str(item)]]))

        return Uia / (Ua * Ia)

    def recommend(self,user):
        # 推荐
        user_result = {}
        item_list = self.get_none_score_item(user)
        for item in item_list:
            user_result[item] = self.cosUI(user,item)
        if self.K is None:
            result = sorted(
                user_result.items(),key=lambda k:k[1],reverse=True
            )
        else:
            result = sorted(
                user_result.items(),key=lambda k:k[1],reverse=True
            )[:self.K]
        print('{} like movies is:'.format(user))
        for i,r in enumerate(result):
            print('top {} is: {}'.format(i+1,self.movie_dict[str(r[0])]))

    def evaluate(self):
        # 评估
        evas = []
        data = pd.read_csv('data/ratings.csv')
        # 随机抽取20个用户
        for user in random.sample([one for one in range(1,6040)],20):
            have_score_items = data[data['UserID']==user]['MovieID'].values
            items = pd.read_csv('data/movies.csv')['MovieID'].values

            user_result = {}
            for item in items:
                user_result[item] = self.cosUI(user,item)
            results = sorted(
                user_result.items(),key=lambda k:k[1],reverse=True
            )[:len(have_score_items)]
            rec_items = []
            for one in results:
                rec_items.append(one[0])
            eva = len(set(rec_items) & set(have_score_items)) / len(have_score_items)
            evas.append(eva)
        return sum(evas) / len(evas)

if __name__ == "__main__":
    cb = CBRecommend(K=10,item_path='data/item.json',user_path='data/user.json')
    cb.recommend(1)
    print('准确率： ',cb.evaluate())


        
