#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

reliable = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
reliable1_dim = pd.read_csv("data/timed-un/reliable1-dim.dat", sep='\t')

""" 
Summary statistics
"""

reliable.groupby(['category']).describe() #.mean()

#plots
# for cat in set(reliable['category']):
#     #plt.figure()
#     #plt.title(cat)
#     cat_df = reliable.loc[reliable['category'] == cat] 
#     plt.plot(cat_df['words'], cat_df['days'], '.')
# plt.show()

""" 
Data pre-processing
"""

# Join releveant data into one dataframe
reg_df = pd.concat([reliable[['perday', 'words']], reliable1_dim], axis=1)

# Convert categorical features into numerical labels
enc = preprocessing.LabelEncoder()
cat = reliable['category']
enc.fit(cat)
reg_df['category'] = enc.transform(cat)

X = reg_df.iloc[:, 1:] # features
y = reg_df.iloc[:, 0]  # objective

manual_split = False
if manual_split == True:

    X_train = X.iloc[:-20]
    y_train = y.iloc[:-20]

    X_test = X.iloc[-20:]
    y_test = y.iloc[-20:]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123) #set random_state for reproducibility

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

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
plt.plot(range(400,2200), range(400,2200), 'k-')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
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
rreg_residuals = y_test - y_pred

#print("Coefficients: \n{}".format(rreg.coef_))
rreg.score(X_test_s, y_test)

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(range(400,2200), range(400,2200), 'k-')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.xlabel('True values')
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
lasso_residuals = y_test - y_pred

#print("Coefficients: \n{}".format(lasso.coef_))

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(range(400,2200), range(400,2200), 'k-')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.xlabel('True values')
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

ols_residuals.describe()
