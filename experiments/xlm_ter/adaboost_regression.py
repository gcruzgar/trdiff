import numpy as np 
import pandas as pd 

import torch
from scripts.utils import load_embeddings, remove_outliers

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split

import scipy.stats as stats
import matplotlib.pyplot as plt

ter = pd.read_csv("data/en-fr-100/en-fr-100-mt_score.txt", sep='\n', header=None)
ter.columns=['score']

xlm_path = "data/en-fr-100/xlm-embeddings/"
features = load_embeddings(xlm_path)

# Use non-zero biber dimensions as features
use_biber=False
if use_biber == True:
    biber = pd.read_csv("data/en-fr-100/en-fr-100.dim", sep='\t')

    drop_cols = biber.columns[(biber == 0).sum() > 0.5*biber.shape[0]]
    biber.drop(drop_cols, axis=1, inplace=True)

    features = features.merge(biber, left_index=True, right_index=True)

# Join data into single dataframe
df = ter.merge(features, left_index=True, right_index=True)

# Remove outliers
rm_out=False
if rm_out == True:
    df = remove_outliers(df, 'score', lq=0.05, uq=0.95) 
    print("data points below 0.05 or above 0.95 quantiles removed")

# Split data into training and tests sets, set random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["score"]), df["score"], test_size=0.2, random_state=42)

# Create model
abr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6), n_estimators=50, learning_rate=0.75, loss='linear', random_state=42)

# Fit model to train data
abr.fit(X_train, y_train)

# Predict and evaluate results
y_pred = abr.predict(X_test)
print("r2-score: %0.3f" % abr.score(X_test, y_test))
print("MSE: %0.3f" % mean_squared_error(y_test, y_pred))

# Quantile-Quantile residual plots
residuals = y_test - y_pred
res = stats.probplot(residuals, plot=plt)
plt.ylabel("Residuals")
plt.title("Normal Probability Plot - Adaboost Decision Tree")
plt.show()