#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn import preprocessing

reliable = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
reliable1_dim = pd.read_csv("data/timed-un/reliable1-dim.dat", sep='\t')

# summary statistics
reliable.groupby(['category']).describe() #.mean()

#plots
# for cat in set(reliable['category']):
#     #plt.figure()
#     #plt.title(cat)
#     cat_df = reliable.loc[reliable['category'] == cat] 
#     plt.plot(cat_df['words'], cat_df['days'], '.')
# plt.show()

# prepare data for linear regression
reg_df = pd.concat([reliable['perday'], reliable1_dim], axis=1)

X_train = reg_df.iloc[:-20, 1:]
y_train = reg_df.iloc[:-20, 0]

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

X_test = reg_df.iloc[-20:, 1:]
y_test = reg_df.iloc[-20:, 0]

X_test_s = scaler.transform(X_test)

""" 
Linear regression - ordinary least squares
"""

ols = linear_model.LinearRegression()
ols.fit(X_train_s, y_train)

y_pred = ols.predict(X_test_s)
ols_residuals = y_test - y_pred

#print("Coefficients: \n{}".format(ols.coef_))
ols.score(X_test_s, y_test)

plt.scatter(y_test, y_pred)
plt.plot(range(400,1800), range(400,1800), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Ordinary Least Squares')
#plt.show()

"""
Linear Regression - Ridge Regresion
""" 

rreg = linear_model.Ridge(alpha=1)
rreg.fit(X_train_s, y_train)

y_pred = rreg.predict(X_test_s)

#print("Coefficients: \n{}".format(rreg.coef_))
rreg.score(X_test_s, y_test)

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(range(400,1800), range(400,1800), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title("Ridge Regression")
#plt.show()

"""
Linear Regression - Lasso
"""

lasso = linear_model.Lasso(alpha=1)
lasso.fit(X_train_s, y_train)

y_pred = lasso.predict(X_test_s)

#print("Coefficients: \n{}".format(lasso.coef_))

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(range(400,1800), range(400,1800), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title("Lasso")
#plt.show()

"""
Scores
"""
print("Ordinary Least Squares: %.3f" % ols.score(X_test_s, y_test))
print("Ridge Regression: %.3f" % rreg.score(X_test_s, y_test))
print("Ridge Regression: %.3f" % lasso.score(X_test_s, y_test))

#Residuals
plt.figure()
plt.plot(y_test, ols_residuals, '.')
plt.xlabel("Real value")
plt.ylabel("Residual")
plt.title("OLS Residuals")
plt.show()
