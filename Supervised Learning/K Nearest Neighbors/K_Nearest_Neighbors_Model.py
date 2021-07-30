import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

dataset = {'y':[[1,2],[2,3],[3,1]], 'b':[[6,5],[7,7],[8,6]]}
new_data_point = [8,9]

def K_Nearest_Neighbors(data, predict, k=5):
    if len(data) >= k:
        warnings.warn("K is set to a value less than total voting groups!")
    distances = []
    [[distances.append([np.linalg.norm(np.array(features) - np.array(predict)), group]) for features in data[group]]for group in data]
    values = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(values).most_common(1)[0][0]
    return vote_result

#[[plt.scatter(ii[0], ii[1], color=i) for ii in dataset[i]] for i in dataset]
#plt.scatter(new_data_point[0], new_data_point[1])
#plt.show()

# That's how I tried to make The Vote Results!
#value = distances[0][0]
#class_predicted = ''
#for i in distances:
    #if i[0] < value:
        #value = i[0]
        #class_predicted = i[1]
    #else:
        #continue

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size  = .2
train_set  = {2:[], 4:[]}
test_set   = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data  = full_data[-int(test_size*len(full_data)):]

[train_set[i[-1]].append(i[:-1]) for i in train_data]
[test_set[i[-1]].append(i[:-1]) for i in test_data]

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = K_Nearest_Neighbors(train_set, data, k=25)
        if vote == group:
            correct += 1
        total += 1

print("Accuracy: ", correct/total)