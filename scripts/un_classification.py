#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from utils import remove_outliers

from sklearn import preprocessing
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

# Load data
timed = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
biber = pd.read_csv("data/timed-un/reliable1-dim.dat", sep='\t')

# Join data into a single dataframe
df = pd.concat([timed['perday'], biber], axis=1)

# Remove outliers
df = remove_outliers(df, 'perday', lq=0.05, uq=0.95) 

# Change regression problem into classification
n_class = 3
if n_class == 3:
    df["class"] = 1 # average 
    df.loc[df["perday"] >= df["perday"].quantile(0.67), "class"] = 0 # easy
    df.loc[df["perday"] <= df["perday"].quantile(0.33), "class"] = 2 # hard
else:
    df["class"] = 1 # easy 
    df.loc[df["perday"] > df["perday"].quantile(0.75), "class"] = 0 # very easy
    df.loc[df["perday"] <= df["perday"].quantile(0.25), "class"] = 3 # very hard
    df.loc[(df["perday"] > df["perday"].quantile(0.25)) & (df["perday"] <= df["perday"].quantile(0.5)), "class"] = 2 # hard

#print("Number of documents per class: \n{}".format(df["class"].value_counts(sort=False).to_string()))

# Split data into training and tests sets, set random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["perday", "class"]), df["class"], test_size=0.15, random_state=42)

# Create classifier
clf_type = 'kn'
if clf_type == 'kn':
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=4, algorithm='auto')

elif clf_type == 'dt':
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()

elif clf_type == 'sv':
    from sklearn.svm import SVC
    clf = SVC(gamma='auto', kernel='linear', C=10.0, probability=True)

# Fit classifier to train data
clf.fit(X_train, y_train)

# Predict using test data
y_pred = clf.predict(X_test)
y_pred_prob = pd.DataFrame(clf.predict_proba(X_test)).round(2)
if n_class == 3:
    y_pred_prob.columns = ["prob 0", "prob 1", "prob 2"]
    diff = {"easy": 0, "average": 1, "difficult": 2}
else:
    y_pred_prob.columns = ["prob 0", "prob 1", "prob 2", "prob 3"]
    diff = {"very easy": 0, "easy": 1, "hard": 2, "very hard": 3}

# Evaluate results
y_res = pd.DataFrame(y_pred, columns=['y_pred'])
y_res['y_test'] = y_test.values

for key in diff.keys():
    
    key_val = y_res.loc[y_res["y_pred"] == diff[key]]
    print( "Accuracy for %s: %0.2f%%" % ( key, accuracy_score( key_val["y_test"], key_val["y_pred"] ) * 100 ) )

save_outputs = False
if save_outputs == True:
    out_df = pd.concat([y_res, y_pred_prob], axis=1)
    out_df.to_csv("un_difficulty_classification.csv", index=None)    
