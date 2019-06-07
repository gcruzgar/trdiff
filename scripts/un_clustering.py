#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn import metrics 
from sklearn.cluster import KMeans 

un_df = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
features = pd.read_csv("data/timed-un/reliable1-dim.dat", sep='\t')

# number of clusters
n = 3

# Clustering based on biber dimensions
estimator = KMeans(n_clusters = n)
y_class = estimator.fit_predict(features)

cluster_coef = estimator.cluster_centers_

y = un_df['perday']
y = pd.concat([y, pd.Series(y_class)], axis=1)
y.columns = ['perday', 'class']
df = pd.concat([y, features], axis=1)

# y_pred=[]
# for i in df.index:
#     y_pred.append(sum( np.array(cluster_coef[df['class'][i]]) * np.array(df.iloc[i,2:]) ))

y_results = pd.DataFrame(y.groupby(['class']).mean())
y_results['stdev'] = y.groupby(['class']).std()

y_results.loc['all'] = [y['perday'].mean(), y['perday'].std()]

print(y_results.round(2))

"""
Classifying on present categories
"""
show_cat=False
if show_cat == True:
    y2 = un_df['perday']
    y_cat = un_df['category']

    y = pd.concat([y2, pd.Series(y_cat)], axis=1)

    y2_results = pd.DataFrame(y2.groupby(['category']).mean())
    y2_results['stdev'] = y2.groupby(['category']).std()

    y2_results.loc['all'] = [y2['perday'].mean(), y2['perday'].std()]

    print(y2_results.round(2))