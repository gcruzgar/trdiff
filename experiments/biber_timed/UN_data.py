#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 

reliable = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
reliable1_dim = pd.read_csv("data/timed-un/reliable1-dim.dat", sep='\t')

""" 
Summary statistics
"""

#reliable.groupby(['category']).describe() #.mean()

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
# can either predict time ('days') or rate ('perday')
target = 'perday' 
if target == 'perday':
    units = 'words per day'
    min_lim, max_lim = 400, 2200 # best fit line limits
else:
    units = target
    min_lim, max_lim = 0, 20

# Join releveant data into one dataframe
reg_df = pd.concat([reliable[[target, 'words']], reliable1_dim], axis=1)

# Convert categorical features into numerical labels
use_cat = False
if use_cat == True:
    enc = preprocessing.LabelEncoder()
    cat = reliable['category']
    enc.fit(cat)
    reg_df['category'] = enc.transform(cat)

# Drop columns with a large number of zeros:
drop_zero_cols = False
if drop_zero_cols == True:
    drop_cols = reg_df.columns[(reg_df == 0).sum() > 0.5*reg_df.shape[0]]
    reg_df.drop(drop_cols, axis=1, inplace=True)

# Plots of each feature against words per day:
# for i in range(2,reg_df.shape[1]):
#     plt.figure()
#     plt.scatter(reg_df['perday'], reg_df.iloc[:,i])
#     plt.title(reg_df.columns[i])
#     plt.xlabel("Words translated per day")
# plt.show()

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
plt.plot(range(min_lim,max_lim), range(min_lim,max_lim), 'k-')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.xlabel('True values (%s)' % units)
plt.ylabel('Predicted values (%s)' % units)
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
plt.plot(range(min_lim,max_lim), range(min_lim,max_lim), 'k-')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.xlabel('True values (%s)' % units)
plt.ylabel('Predicted values (%s)' % units)
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
plt.plot(range(min_lim,max_lim), range(min_lim,max_lim), 'k-')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.xlabel('True values (%s)' % units)
plt.ylabel('Predicted values (%s)' % units)
plt.title("Lasso")
#plt.show()

"""
Scores
"""
print("Ordinary Least Squares: %.3f" % ols.score(X_test_s, y_test))
print("Ridge Regression: %.3f" % rreg.score(X_test_s, y_test))
print("Lasso Regression: %.3f" % lasso.score(X_test_s, y_test))

#Residuals
plt.figure()
plt.plot(y_test, ols_residuals, '.')
plt.xlabel("Real value")
plt.ylabel("Residual")
plt.title("OLS Residuals")
#plt.show()

#ols_residuals.describe()

"""
Predict time taken
"""
if target == 'days':
    lasso = linear_model.Lasso(alpha=1)
    lasso.fit(X_train_s, y_train)

    y_pred = lasso.predict(X_test_s)
    lasso_residuals = y_test - y_pred

    plt.figure()
    r = plt.scatter(X_test['words'], y_test)
    p = plt.scatter(X_test['words'], y_pred)
    plt.plot(np.unique(X_test['words']), np.poly1d(np.polyfit(X_test['words'], y_pred, 1))(np.unique(X_test['words'])), 'r--')
    plt.plot(np.unique(X_test['words']), np.poly1d(np.polyfit(X_test['words'], y_test, 1))(np.unique(X_test['words'])), 'k--')
    plt.xlabel('Document length (words)')
    plt.ylabel('Time taken to Translate (days)')
    plt.legend((p, r), ('Predicted', 'Real'))

plt.show()

""" 
Plot time taken vs document lenght with MAE lines
"""
def time_length_plot(df=reliable):

    # Line of best fit and distance from each point to the line

    plt.figure()

    c, m = np.polynomial.polynomial.polyfit(df['words'], df['days'], 1)
    y_pred = m*df['words'] + c 

    residuals = df['days'] - y_pred
    #median_error = abs(residuals).median()
    MAE = mean_absolute_error(df['days'], y_pred) # mean absolute error

    plt.plot(df['words'], df['days'], '.')
    plt.plot(df['words'],y_pred, 'k--')

    x1 = np.linspace(min(df['words']), max(df['words']))
    y1 = m*x1 + c

    plt.plot(x1, y1+MAE, 'r--') # confidence intervals (bestfit +/- MAE)
    plt.plot(x1, y1-MAE, 'r--')

    #pos_res = residuals.loc[residuals >  MAE] # points above the line
    #neg_res = residuals.loc[residuals < -MAE] # points below the line

    plt.xlabel("Document length (words)")
    plt.ylabel("Time taken to translate (days)")

    plt.show()

time_length_plot(df=reliable)