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
#plt.plot(range(400,2200), range(400,2200), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('Ordinary Least Squares')
plt.show()
