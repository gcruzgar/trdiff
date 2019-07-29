#!/usr/bin/env python3
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score 
from utils import linear_regression

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

features = pd.read_csv("data/wto/biber_wto.dat", sep='\t')
tr = pd.read_csv("data/wto/wto_timed.csv")

df = pd.concat([features, tr], axis=1)
df = remove_outliers(df, q=10)

y_fr = df["PERDAY FRENCH"]
y_sp = df["PERDAY SPANISH"]

X = df.iloc[:, :-7]

"""
French only
"""

y = y_fr

ols, scaler, X_test, y_test = linear_regression(X, y, test_size=0.1, random_state=210, plots=False)

X_test_s = scaler.transform(X_test)
y_pred = ols.predict(X_test_s)
ols_residuals = y_test - y_pred

#print("Coefficients: \n{}".format(ols.coef_))
print("French only")
print(ols.score(X_test_s, y_test))

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(range(100, 5000), range(100, 5000), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('OLS - French')

"""
Spanish only
"""

y = y_sp

ols, scaler, X_test, y_test = linear_regression(X, y, test_size=0.1, random_state=210, plots=False)

X_test_s = scaler.transform(X_test)
y_pred = ols.predict(X_test_s)
ols_residuals = y_test - y_pred

#print("Coefficients: \n{}".format(ols.coef_))
print("\nSpanish only")
print(ols.score(X_test_s, y_test))

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(range(100, 5000), range(100, 5000), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('OLS - Spanish')

# #Residuals
# plt.figure()
# plt.plot(y_test, ols_residuals, '.')
# plt.xlabel("Real value")
# plt.ylabel("Residual")
# plt.title("OLS Residuals - Spanish")
# plt.show()

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
print("\nTrain with French, test with Spanish")
print(ols.score(X_test_s, y_test))

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(range(100, 5000), range(100, 5000), 'k-')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('OLS - Train French, test Spanish')
#plt.show()

""" 
Train with French, test with Spanish
"""

X_train = X
X_test = X
y_train = y_sp
y_test = y_fr

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

X_test_s = scaler.transform(X_test)

ols = linear_model.LinearRegression()
ols.fit(X_train_s, y_train)

y_pred = ols.predict(X_test_s)
ols_residuals = y_test - y_pred

#print("Coefficients: \n{}".format(ols.coef_))
print("\nTrain with Spanish, test with French")
print(ols.score(X_test_s, y_test))

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(range(100, 5000), range(100, 5000), 'k-')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('OLS - Train Spanish, test French')
#plt.show()

"""
Combine French and Spanish
"""

X_combined = pd.concat([X, X], axis=0, ignore_index=True)
y_combined = pd.concat([y_fr, y_sp], axis=0, ignore_index=True)

ols, scaler, X_test, y_test = linear_regression(X_combined, y_combined, test_size=0.2, random_state=210, plots=False)

X_test_s = scaler.transform(X_test)
y_pred = ols.predict(X_test_s)
ols_residuals = y_test - y_pred

plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(range(0, 6000), range(0, 6000), 'k-')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('OLS - French and Spanish')

# #Residuals
# plt.figure()
# plt.plot(y_test, ols_residuals, '.')
# plt.xlabel("Real value")
# plt.ylabel("Residual")
# plt.title("OLS Residuals - French and Spanish")
plt.show()

print("\nFrench and Spanish combined")
print(ols.score(X_test_s, y_test))
