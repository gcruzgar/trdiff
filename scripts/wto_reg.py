#!/usr/bin/env python3
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def remove_outliers(time_df, n):
    """ Only keep cases where the time taken is less than n times as long for one language compared to the other.
    Also drop cases where the difference in days is larger than 8*n. (20 days for n=2.5)"""

    time_df = time_df.loc[(time_df["DAYS FRENCH"] / time_df["DAYS SPANISH"]) < n]
    time_df = time_df.loc[(time_df["DAYS SPANISH"] / time_df["DAYS FRENCH"]) < n] 
    time_df = time_df.loc[abs(time_df["DAYS FRENCH"] - time_df["DAYS SPANISH"]) < 8*n]

    return time_df

features = pd.read_csv("data/biber_wto.dat", sep='\t')
tr = pd.read_csv("data/wto_edited.csv")

df = pd.concat([features, tr], axis=1)
df = remove_outliers(df, n=2.5)

y_fr = df["PERDAY FRENCH"]
y_sp = df["PERDAY SPANISH"]

X = df.iloc[:, :-7]
y = y_fr

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=420) #set random_state for reproducibility

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
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Ordinary Least Squares')
#plt.show()

#Residuals
plt.figure()
plt.plot(y_test, ols_residuals, '.')
plt.xlabel("Real value")
plt.ylabel("Residual")
plt.title("OLS Residuals")
plt.show()

""" 
Train with French, test with Spanish
"""

X_train = X
X_test = X
y_train = y_fr
y_test = y_sp

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

X_test_s = scaler.transform(X_test)

ols = linear_model.LinearRegression()
ols.fit(X_train_s, y_train)

y_pred = ols.predict(X_test_s)
ols_residuals = y_test - y_pred

#print("Coefficients: \n{}".format(ols.coef_))
ols.score(X_test_s, y_test)

plt.scatter(y_test, y_pred)
#plt.plot(range(400,2200), range(400,2200), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Ordinary Least Squares')
plt.show()

"""
Combine French and Spanish
"""

X_combined = pd.concat([X, X], axis=0, ignore_index=True)
y_combined = pd.concat([y_fr, y_sp], axis=0, ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.10, random_state=210) #set random_state for reproducibility

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

X_test_s = scaler.transform(X_test)

ols = linear_model.LinearRegression()
ols.fit(X_train_s, y_train)

y_pred = ols.predict(X_test_s)
ols_residuals = y_test - y_pred

plt.scatter(y_test, y_pred)
#plt.plot(range(100,5000), range(100,5000), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Ordinary Least Squares')
#plt.show()

#Residuals
plt.figure()
plt.plot(y_test, ols_residuals, '.')
plt.xlabel("Real value")
plt.ylabel("Residual")
plt.title("OLS Residuals")
plt.show()

ols.score(X_test_s, y_test)