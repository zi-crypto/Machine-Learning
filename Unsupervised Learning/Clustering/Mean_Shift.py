import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import MeanShift
from sklearn import preprocessing
style.use('ggplot')
import numpy as np

import pandas as pd
from Handling_Non_Numeric_Data import handle_non_numerical_data
colors = 10*["g","r","c","b","k"]


df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)
df.drop(['body','name'], 1, inplace=True)
print(df.head())
df.fillna(0,inplace=True)

handle_non_numerical_data(df)

df.drop(['ticket','home.dest'], 1, inplace=True)


X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survived_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[ (original_df['cluster_group'] == float(i)) ]
    survived_cluster = temp_df[ (temp_df['survived'] == 1) ]
    survived_rate = len(survived_cluster)/len(temp_df)
    survived_rates[i] = survived_rate

print(survived_rates) #!!!!!!!!!!!!!!!!!!!!!!!!! What !!!!!!!!!!!!!!!!!!!!!!!!!!