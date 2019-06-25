#!/usr/bin/env python3

import torch
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def linear_regression(X, y, test_size=0.2, random_state=123, plots=True):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state) #set random_state for reproducibility

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    X_test_s = scaler.transform(X_test)

    ols = linear_model.LinearRegression()
    ols.fit(X_train_s, y_train)

    y_pred = ols.predict(X_test_s)
    ols_residuals = y_test - y_pred

    if plots == True:

        # Real vs Predicted
        plt.figure()
        plt.scatter(y_test, y_pred)
        #plt.plot(range(0, 1), range(0, 1), 'k-')
        plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
        plt.xlabel('True values')
        plt.ylabel('Predicted values')
        plt.title('OLS - Real vs Predicted')

        # Residuals
        plt.figure()
        plt.plot(y_test, ols_residuals, '.')
        plt.xlabel("Real value")
        plt.ylabel("Residual")
        plt.title("OLS Residuals")
        plt.show()

    #print("Coefficients: \n{}".format(ols.coef_))
    print("r2-score: %0.4f" % ols.score(X_test_s, y_test))

    return ols, scaler, X_test, y_test

def remove_outliers(df, filter_var, lq=0.1, uq=None):
    """ 
    Remove values above upper quantile, uq, and below lower quantile, lq, from dataframe, df, based on column, filter_var.  
    """

    if uq > 1 or lq > 1:
        lq /= 100
        uq /= 100
        print("percentiles should all be in the interval [0, 1]; %.2f and %.2f used instead." % (lq, uq))

    if uq == None:
        uq = 1 - lq

    r_uq = df[filter_var].quantile(q=uq)
    r_lq = df[filter_var].quantile(q=lq)
    df = df.loc[df[filter_var] < r_uq]
    df = df.loc[df[filter_var] > r_lq]

    return df

def load_embeddings(xlm_path = "data/xlm-embeddings/"):

    embeddings = pd.concat([
    pd.DataFrame(torch.load(xlm_path+"xlm-embeddings-0_499.pt").data.numpy(), index=range(0,500)),
    pd.DataFrame(torch.load(xlm_path+"xlm-embeddings-500_999.pt").data.numpy(), index=range(500,1000)),
    pd.DataFrame(torch.load(xlm_path+"xlm-embeddings-1500_1999.pt").data.numpy(), index=range(1500,2000)),
    pd.DataFrame(torch.load(xlm_path+"xlm-embeddings-2000_2499.pt").data.numpy(), index=range(2000,2500)),
    pd.DataFrame(torch.load(xlm_path+"xlm-embeddings-2500_2999.pt").data.numpy(), index=range(2500,3000)),
    pd.DataFrame(torch.load(xlm_path+"xlm-embeddings-3000_3499.pt").data.numpy(), index=range(3000,3500)),
    pd.DataFrame(torch.load(xlm_path+"xlm-embeddings-3500_3999.pt").data.numpy(), index=range(3500,4000))
    ], axis=0)

    #pd.DataFrame(torch.load(xlm_path+"xlm-embeddings-1000_1499.pt").data.numpy(), index=range(1000,1500)),
    
    return embeddings
