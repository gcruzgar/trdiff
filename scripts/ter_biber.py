#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

features = pd.read_csv("data/en-fr-100.dim", sep='\t')
ter = pd.read_csv("data/en-fr-100-mt_score.txt", sep='\t', header=None)
ter.columns=['Score']

df = pd.concat([features, ter], axis=1) 

drop_outliers = True
if drop_outliers == True:
    r_95p = df['Score'].quantile(q=0.95)
    #r_5p = df['Score'].quantile(q=0.05)
    df = df.loc[df['Score'] < r_95p]
    #df = df.loc[df['Score'] > r_5p]

X = df.drop(columns=['Score'])
y = df['Score']

from utils import linear_regression

ols, scaler = linear_regression(X, y)
