import numpy as np 
import pandas as pd 

import torch
from scripts.utils import load_embeddings, remove_outliers

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

ter = pd.read_csv("data/en-fr-100-mt_score.txt", sep='\n', header=None)
ter.columns=['score']

xlm_path = "data/xlm-embeddings/"
features = load_embeddings(xlm_path)

# Join data into single dataframe
df = ter.merge(features, left_index=True, right_index=True)

# Remove outliers
rm_out=False
if rm_out == True:
    df = remove_outliers(df, 'score', lq=0.05, uq=0.95) 
    print("data points below 0.05 or above 0.95 quantiles removed")

# Classify scores based on percentile
df["class"] = 1 # average translation
df.loc[df["score"] >= df["score"].quantile(0.67), "class"] = 0 # good translation
df.loc[df["score"] <= df["score"].quantile(0.33), "class"] = 2 # bad translation

# Split data into training and tests sets, set random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["score", "class"]), df["class"], test_size=0.2, random_state=42)

# Create classifier
clf = AdaBoostClassifier(algorithm='SAMME', n_estimators=50, learning_rate=1.0)

# Fit classifier to train data
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate results
print(clf.score(X_test, y_test).round(3))