import pandas as pd 
import numpy as np 
import torch
import matplotlib.pyplot as plt 

from scripts.utils import load_embeddings, remove_outliers, linear_regression

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

ter = pd.read_csv("data/en-fr-100-mt_score.txt", sep='\n', header=None)
ter.columns=['score']

xlm_path = "data/xlm-embeddings/"
features = load_embeddings(xlm_path)

# Join data into single dataframe
df = ter.merge(features, left_on=ter.index, right_on=features.index)

# Remove outliers
rm_out=True
if rm_out == True:
    df = remove_outliers(df, 'score', lq=0.05, uq=0.95) 
    print("data points below 0.05 or above 0.95 quantiles removed")

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["score"]), df["score"], test_size=0.2, random_state=42)

# for C in [0.00001, 0.0001, 0.001, 0.01, 0.1, 10, 100]:

#     # Create classifier
#     svr = SVR(C=C, kernel='rbf', gamma='scale')

#     # Fit classifier to train data
#     svr.fit(X_train, y_train)

#     # Evaluate results
#     print("C = %f, score=%0.3f\n" % (C, svr.score(X_test, y_test)))

# Create classifier
C=10
svr = SVR(C=C, kernel='rbf', gamma='scale')

# Fit classifier to train data
svr.fit(X_train, y_train)

# Predict and evaluate results
print("C = %0.3f, score=%0.3f\n" % (C, svr.score(X_test, y_test)))
