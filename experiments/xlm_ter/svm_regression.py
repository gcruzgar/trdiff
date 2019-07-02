import torch
import numpy as np
import pandas as pd  

from scripts.utils import load_embeddings, remove_outliers, linear_regression

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split

import scipy.stats as stats
import matplotlib.pyplot as plt

ter = pd.read_csv("data/en-fr-100/en-fr-100-mt_score.txt", sep='\n', header=None)
ter.columns=['score']

xlm_path = "data/en-fr-100/xlm-embeddings/"
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

#     # Create model
#     svr = SVR(C=C, kernel='rbf', gamma='scale')

#     # Fit model to train data
#     svr.fit(X_train, y_train)

#     # Evaluate results
#     print("C = %f, score=%0.3f\n" % (C, svr.score(X_test, y_test)))

# Create model
C=10
svr = SVR(C=C, kernel='rbf', gamma='scale')

# Fit model to train data
svr.fit(X_train, y_train)

# Predict and evaluate results
y_pred = svr.predict(X_test)
print("\nC = %0.3f") 
print("r2-score: %0.3f" % svr.score(X_test, y_test))
print("MSE: %0.3f" % mean_squared_error(y_test, y_pred))

# Quantile-Quantile residual plots
residuals = y_test - y_pred
res = stats.probplot(residuals, plot=plt)
plt.ylabel("Residuals")
plt.title("Normal Probability Plot - SVM")
plt.show()