#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from scripts.utils import remove_outliers, kfold_crossvalidation

from sklearn import preprocessing
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load data
timed = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
biber = pd.read_csv("data/timed-un/reliable1-dim.dat", sep='\t')

encode_category = False
if encode_category == True:

    enc = preprocessing.LabelEncoder()
    cat = timed['category']
    enc.fit(cat)
    timed['cat'] = enc.transform(cat)

# Join data into a single dataframe
df = pd.concat([timed['perday'], biber], axis=1)

# Remove outliers
df = remove_outliers(df, 'perday', lq=0.05, uq=0.95)
df.reset_index(inplace=True) 

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

def classification(df, clf_type):
    # Split data into training and tests sets, set random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["perday", "class"]), df["class"], test_size=0.15, random_state=42)

    # Create classifier
    if clf_type == 'kn':
        print("Generating nearest kneighbour classifier...")
        clf = KNeighborsClassifier(n_neighbors=4, algorithm='auto')

    elif clf_type == 'dt':
        print("Generating decision tree classifier...")
        clf = DecisionTreeClassifier()

    elif clf_type == 'sv':
        print("Generating support vector machine classifier...")
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
    y_df = pd.DataFrame(y_pred, columns=['y_pred'])
    y_df['y_test'] = y_test.values

    return y_df

# Use k-fold cross-validation
def crossvalidation(df):
    
    X = df.drop(columns=["perday", "class"])
    y = df["class"]
    y_df = kfold_crossvalidation(X, y, k=12, method='clf', output='df')

    return y_df

#y_df = classification(df, clf_type='sv')
y_df = crossvalidation(df)

diff = {"fast translation": 0, "average translation": 1, "slow translation": 2}
print("\nclassification report:\n")
print(classification_report(y_df['y_test'], y_df['y_pred'], target_names=diff))

save_outputs = False
if save_outputs == True:
    out_df = pd.concat([y_res, y_pred_prob], axis=1)
    out_df.to_csv("un_difficulty_classification.csv", index=None)    
