#!/usr/bin/env python3

import numpy as np
import pandas as  pd 
import matplotlib.pyplot as plt  
from sklearn.svm import LinearSVR
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

""" TER """
features = pd.read_csv("data/en-fr-100/en-fr-100.dim", sep='\t')
ter = pd.read_csv("data/en-fr-100/en-fr-100-mt_score.txt", sep='\t', header=None)
ter.columns=['Score']

df = pd.concat([features, ter], axis=1) 

drop_outliers = True
if drop_outliers == True:
    r_97p = df['Score'].quantile(q=0.97)
    r_3p = df['Score'].quantile(q=0.03)
    df = df.loc[df['Score'] < r_97p]
    df = df.loc[df['Score'] > r_3p]

X = df.drop(columns=['Score'])
y = df['Score']


# """ Words Translated Per Day """
# reliable = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
# reliable1_dim = pd.read_csv("data/timed-un/reliable1-dim.dat", sep='\t')

# df = pd.concat([reliable[['perday', 'words']], reliable1_dim], axis=1)
# X = df.drop(columns=['perday']) # features
# y = df['perday']  # objective

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=123) #set random_state for reproducibility

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)

X_test_s = scaler.transform(X_test)

lsvr = LinearSVR(random_state=1, tol=1e-4, C=2.0, max_iter=10000)
print("\nFitting SVR...")
lsvr.fit(X_train_s, y_train)

y_pred = lsvr.predict(X_test_s)
lsvr_residuals = y_test - y_pred

# Real vs Predicted
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), 'k-')
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.title('LinearSVR- Real vs Predicted')

#Residuals
plt.figure()
plt.plot(y_test, lsvr_residuals, '.')
plt.xlabel("Real value")
plt.ylabel("Residual")
plt.title("LinearSVR Residuals")
plt.show()

#print("Coefficients: \n{}".format(lsvr.coef_))
print("\nr2-score: %0.4f" % lsvr.score(X_test_s, y_test))