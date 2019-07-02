#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from scripts.utils import remove_outliers

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

features = pd.read_csv("data/en-fr-100/en-fr-100.dim", sep='\t')
ter = pd.read_csv("data/en-fr-100/en-fr-100-mt_score.txt", sep='\t', header=None)
ter.columns=['score']

df = pd.concat([features, ter], axis=1) 

# Remove outliers
df = remove_outliers(df, 'score', lq=0.05, uq=0.95) 

""" Regression """
from scripts.utils import linear_regression

X = df.drop(columns=['score'])
y = df['score']

ols, scaler, X_test, y_test = linear_regression(X, y)

""" Classification """
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier

# Classify scores depedning on percentile
df["class"] = 1 # average 
df.loc[df["score"] >= df["score"].quantile(0.67), "class"] = 0 # good
df.loc[df["score"] <= df["score"].quantile(0.33), "class"] = 2 # bad

#print("Number of documents per class: \n{}".format(df["class"].value_counts(sort=False).to_string()))

# Split data into training and tests sets, set random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["score", "class"]), df["class"], test_size=0.2, random_state=42)

# Create classifier
neigh = KNeighborsClassifier(n_neighbors=9, algorithm='auto')

# Fit classifier to train data
neigh.fit(X_train, y_train)

# Predict using test data
y_pred = neigh.predict(X_test)
y_pred_prob = pd.DataFrame(neigh.predict_proba(X_test)).round(2)
y_pred_prob.columns = ["prob 0", "prob 1", "prob 2"]

# Evaluate results
diff = {"good translation": 0, "average translation": 1, "bad translation": 2}

y_res = pd.DataFrame(y_pred, columns=['y_pred'])
y_res['y_test'] = y_test.values

for key in diff.keys():
        
    key_val = y_res.loc[y_res["y_pred"] == diff[key]]
    print( "Accuracy for %s: %0.2f%%" % ( key, accuracy_score( key_val["y_test"], key_val["y_pred"] ) * 100 ) )
