import math 
class ItemCF:
    def __init__(self):
        self.user_score_dict = self.initUserScore()
        self.items_sim = self.ItemSimilarity()

    def initUserScore(self):
        user_score_dict = {"A": {"a": 3.0, "b": 4.0, "c": 0.0, "d": 3.5, "e": 0.0},
                           "B": {"a": 4.0, "b": 0.0, "c": 4.5, "d": 0.0, "e": 3.5},
                           "C": {"a": 0.0, "b": 3.5, "c": 0.0, "d": 0., "e": 3.0},
                           "D": {"a": 0.0, "b": 4.0, "c": 0.0, "d": 3.50, "e": 3.0}}
        return user_score_dict

    def ItemSimilarity(self):
        # 计算物品之间相似度
        itemSim = dict()

        # 物品有多少用户发生过行为
        item_user_count = dict()
        # 俩物品被同一个用户评价过
        count = dict()
        for user,item in self.user_score_dict.items():
            for i in item.keys():
                item_user_count.setdefault(i,0)
                if self.user_score_dict[user][i] > 0.0:
                    item_user_count[i] += 1
                for j in item.keys():
                    count.setdefault(i,{}).setdefault(j,0)
                    if self.user_score_dict[user][i] > 0 and self.user_score_dict[user][j] > 0 and i != j:
                        count[i][j] += 1

        # 相似矩阵计算
        for i, related_items in count.items():
            itemSim.setdefault(i,dict()) 
            for j,cuv in related_items.items():
                itemSim[i].setdefault(j,0)
                itemSim[i][j] = cuv / math.sqrt(item_user_count[i] * item_user_count[j])
        return itemSim


    def preUserItemScore(self,userA,item):
        # 预测用户对item的评分
        score = 0.0 
        for item1 in self.items_sim[item].keys():
            if item1 != item:
                score += (self.items_sim[item][item1] * self.user_score_dict[userA][item1])
        return score

    def recommend(self,userA):
        user_item_score_dict = dict()
        for item in self.user_score_dict[userA].keys():
            user_item_score_dict[item] = self.preUserItemScore(userA,item)
        return user_item_score_dict

if __name__ == "__main__":
    ib = ItemCF()
    print(ib.recommend("C"))