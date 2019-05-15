#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

features = pd.read_csv("data/biber_mt.dat", sep='\t')
score = pd.read_csv("data/mt_score.dat", sep = ' ', header=None, usecols=[1])
score.columns= ['score']

reg_df = pd.concat([features, score], axis=1)

# Drop columns with a large number of zeros:
# drop_cols = reg_df.columns[(reg_df == 0).sum() > 0.5*reg_df.shape[1]]
# reg_df.drop(drop_cols, axis=1, inplace=True)

X = reg_df.drop(columns=['score']) # features
y = reg_df['score']  # objective

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1234) #set random_state for reproducibility

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

X_test_s = scaler.transform(X_test)

ols = linear_model.LinearRegression()
ols.fit(X_train_s, y_train)

y_pred = ols.predict(X_test_s)
ols_residuals = y_test - y_pred

plt.scatter(y_test, y_pred)
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.plot(np.linspace(0.2,0.8,7), np.linspace(0.2,0.8,7), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Ordinary Least Squares')

#Residuals
plt.figure()
plt.plot(y_test, ols_residuals, '.')
plt.xlabel("Real value")
plt.ylabel("Residual")
plt.title("OLS Residuals")
plt.show()

print("r-score: %0.3f" % (ols.score(X_test_s, y_test)))