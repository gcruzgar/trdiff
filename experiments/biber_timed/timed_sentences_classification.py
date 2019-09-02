#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from scripts.utils import remove_outliers

from sklearn import preprocessing
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

lan = "es"
b_lan="en"
# Load data
timed = pd.read_csv("data/un-timed-sentences/en-"+lan+".processed", sep='\t')
biber = pd.read_csv("data/un-timed-sentences/en-"+lan+"-biber."+b_lan, sep='\t')

# Calculated words translated per second
words=[]
for i in timed.index:
    words.append(len(timed['Segment'][i].split()))
timed["persec"] = words / timed['Time-to-edit'] * 100

# Only use informative dimensions
filter_biber = False
if filter_biber == True:
    drop_cols = biber.columns[(biber == 0).sum() > 0.75*biber.shape[0]]
    biber.drop(drop_cols, axis=1, inplace=True)

# Join data into a single dataframe
df = pd.concat([timed['persec'], biber], axis=1)

# Remove outliers
df = remove_outliers(df, 'persec', lq=0.05, uq=0.95) 

# Change regression problem into classification
df["class"] = 1 # average 
df.loc[df["persec"] >= df["persec"].quantile(0.67), "class"] = 0 # easy
df.loc[df["persec"] <= df["persec"].quantile(0.33), "class"] = 2 # hard

# Split data into training and tests sets, set random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["persec", "class"]), df["class"], test_size=0.15, random_state=42)

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

# Evaluate results
y_res = pd.DataFrame(y_pred, columns=['y_pred'])
y_res['y_test'] = y_test.values

from sklearn.metrics import classification_report
diff = {"fast translation": 0, "average translation": 1, "slow translation": 2}

if lan=="es":
    language = "Spanish"
elif lan=="fr":
    language = "French"

print("\nclassification report (%s):\n" % language)
print(classification_report(y_test, y_pred, target_names=diff)) 