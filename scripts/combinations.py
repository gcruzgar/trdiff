#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def remove_outliers(time_df, q):
    """ 
    New: Remove values outside q to (100-q) quantiles.

    Old: Only keep cases where the time taken is less than n times as long for one language compared to the other.
    Also drop cases where the difference in days is larger than 8*n. (20 days for n=2.5)
    """
    # old version
    # # remove outliers based on ratio and minimum difference
    # time_df = time_df.loc[(time_df["DAYS FRENCH"] / time_df["DAYS SPANISH"]) < n]
    # time_df = time_df.loc[(time_df["DAYS SPANISH"] / time_df["DAYS FRENCH"]) < n] 
    # time_df = time_df.loc[abs(time_df["DAYS FRENCH"] - time_df["DAYS SPANISH"]) < 8*n]

    # remove outliers based on quantiles
    perday_ratio = time_df['DAYS FRENCH'] / time_df['DAYS SPANISH']
    uq = perday_ratio.quantile(q=(100-q)/100)
    lq = perday_ratio.quantile(q=q/100)
    perday_ratio = perday_ratio.loc[perday_ratio < uq]
    perday_ratio = perday_ratio.loc[perday_ratio > lq]
    time_df = time_df.loc[perday_ratio.index]

    return time_df

# Load and join UN translation data
un_tr = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
un_dim = pd.read_csv("data/timed-un/reliable1-dim.dat", sep='\t')
un_df = pd.concat([un_tr[['words', 'perday']], un_dim], axis=1)

# Load and join WTO translation data (+ option to filter outliers)
wto_tr = pd.read_csv("data/wto_timed.csv") 
wto_dim = pd.read_csv("data/biber_wto.dat", sep='\t')  
wto_df = pd.concat([wto_tr, wto_dim], axis=1)
wto_df = remove_outliers(wto_df,q=5) 
wto_df = wto_df.drop(columns=['JOB N°', 'SYMBOL', 'DAYS FRENCH', 'DAYS SPANISH', 'PERDAY SPANISH'])

# Combine datasets:
wto_df.rename(index=str, columns={"WORDS": "words", "PERDAY FRENCH": "perday"}, inplace=True)
reg_df = pd.concat([un_df, wto_df], axis=0)

# Drop columns with a large number of zeros:
drop_zero_cols = False
if drop_zero_cols == True:
    drop_cols = reg_df.columns[(reg_df == 0).sum() > 0.5*reg_df.shape[0]]
    reg_df.drop(drop_cols, axis=1, inplace=True)

# Drop outliers:
drop_outliers = True
if drop_outliers == True:
    r_95p = reg_df['perday'].quantile(q=0.95)
    r_5p = reg_df['perday'].quantile(q=0.05)
    reg_df = reg_df.loc[reg_df["perday"] < r_95p]
    reg_df = reg_df.loc[reg_df["perday"] > r_5p]

# Preprocessing:
X = reg_df.drop(columns=['perday']) # features
y = reg_df['perday']  # objective

manual_split = False
if manual_split == True:

    X_train = X.iloc[:-20]
    y_train = y.iloc[:-20]

    X_test = X.iloc[-20:]
    y_test = y.iloc[-20:]
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=920) #set random_state for reproducibility

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

plt.scatter(y_test, y_pred)
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.plot(range(100,2500), range(100,2500), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('OLS - UNOG & WTO')
#plt.show()

#Residuals
plt.figure()
plt.plot(y_test, ols_residuals, '.')
plt.xlabel("Real value")
plt.ylabel("Residual")
plt.title("OLS Residuals - UNOG & WTO")
plt.show()

#print("Coefficients: \n{}".format(ols.coef_))
ols_residuals.describe()
print("r-score: %0.3f" % (ols.score(X_test_s, y_test)))

# test_df = pd.DataFrame(data=X_test_s)
# test_df['y'] = y_test.reset_index(drop=True)

# test_df = test_df.loc[test_df['y'] < 3000]
# ols.score(test_df.drop(columns=['y']), test_df['y'])
