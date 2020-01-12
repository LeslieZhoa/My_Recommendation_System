'''
the code is based on https://github.com/saimadhu-polamuri/CollaborativeFiltering
'''
from recommendation_data import dataset
import math

def person_sim(person1,person2):
    '''
    use cosine to compute the score
    '''
    both_rated = {}
    for key in dataset[person1]:
        if key in dataset[person2]:
            both_rated[key] = 1
    
    num_ratings = len(both_rated)
    if num_ratings == 0:
        return 0

    # compute the cosine
    numerator_value = sum([dataset[person1][key]*dataset[person2][key] for key in both_rated] )
    denominator_value = math.sqrt(sum([math.pow(dataset[person1][key],2) for key in both_rated]))*\
                        math.sqrt(sum([math.pow(dataset[person2][key],2) for key in both_rated]))
    if denominator_value == 0:
        return 0
    
    else:
        return numerator_value/denominator_value


def user_recommen(person):
    totals = {}
    simSums = {}
    
    sim_dict = {}
    for other in dataset:
        if other == person:
            continue
        sim = person_sim(person,other)
        sim_dict[other] = sim
        if sim <= 0:
            continue
        for key in dataset[other]:
            if key not in dataset[person] or dataset[person][key]==0:
                totals[key] = totals.get(key,0) + dataset[other][key]*sim
                simSums[key] = simSums.get(key,0) + sim
    print('similarity list: ')
    for key,value in sim_dict.items():
        print('person: {}|score: {}'.format(key,value))

    rankings = [(total/simSums[key],key) for key,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    
    return rankings

if __name__ == "__main__":
    person = 'Toby'
    rankings = user_recommen(person)
    print('the recommendate movie list: ')
    for (score,key) in rankings:
        print('movie: {}| score: {}'.format(key,score)) 