#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from scripts.utils import remove_outliers

from sklearn import preprocessing
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load data
timed = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
biber = pd.read_csv("data/timed-un/reliable1-dim.dat", sep='\t')

# Join data into a single dataframe
df = pd.concat([timed['perday'], biber], axis=1)

# Remove outliers
df = remove_outliers(df, 'perday', lq=0.05, uq=0.95) 

# Change regression problem into classification
df["class"] = 1 # average 
df.loc[df["perday"] >= df["perday"].quantile(0.67), "class"] = 0 # easy
df.loc[df["perday"] <= df["perday"].quantile(0.33), "class"] = 2 # hard

# Split data into training and tests sets, set random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["perday", "class"]), df["class"], test_size=0.15, random_state=42)

# Create classifier
neigh = KNeighborsClassifier(n_neighbors=10, algorithm='auto')

# Fit classifier to train data
neigh.fit(X_test, y_test)

# Predict using test data
y_pred = neigh.predict(X_test)
y_pred_prob = neigh.predict_proba(X_test)

# Evaluate results
diff = {"easy": 0, "average": 1, "difficult": 2}

y_res = pd.DataFrame(y_pred, columns=['y_pred'])
y_res['y_test'] = y_test.values

for key in diff.keys():
    
    key_val = y_res.loc[y_res["y_pred"] == diff[key]]
    print( "Accuracy for %s: %0.2f%%" % ( key, accuracy_score( key_val["y_test"], key_val["y_pred"] ) * 100 ) )
