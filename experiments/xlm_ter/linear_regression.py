import pandas as pd 
import numpy as np 
import torch

from sklearn.metrics import mean_squared_error 
from scripts.utils import load_embeddings, remove_outliers, linear_regression

import scipy.stats as stats
import matplotlib.pyplot as plt

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
    
ols, scaler, X_test, y_test = linear_regression(df.drop(columns=['score']), df['score'])

y_pred = ols.predict(scaler.transform(X_test))
print("MSE: %0.3f" % mean_squared_error(y_test, y_pred))

# Quantile-Quantile residual plots
residuals = y_test - y_pred
res = stats.probplot(residuals, plot=plt)
plt.ylabel("Residuals")
plt.title("Normal Probability Plot - Linear Regression")
plt.show()