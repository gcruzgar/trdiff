#!/usr/bin/env python3
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def linear_regression(X, y, test_size=0.2, random_state=123):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state) #set random_state for reproducibility

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    X_test_s = scaler.transform(X_test)

    ols = linear_model.LinearRegression()
    ols.fit(X_train_s, y_train)

    y_pred = ols.predict(X_test_s)
    ols_residuals = y_test - y_pred

    # Real vs Predicted
    plt.figure()
    plt.scatter(y_test, y_pred)
    #plt.plot(range(0, 1), range(0, 1), 'k-')
    plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('OLS - Real vs Predicted')

    #Residuals
    plt.figure()
    plt.plot(y_test, ols_residuals, '.')
    plt.xlabel("Real value")
    plt.ylabel("Residual")
    plt.title("OLS Residuals")
    plt.show()

    #print("Coefficients: \n{}".format(ols.coef_))
    print("r2-score: %0.4f" % ols.score(X_test_s, y_test))

    return ols, scaler
