#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, 
#from sklearn import preprocessing

reliable = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
reliable1_dim = pd.read_csv("data/timed-un/reliable1-dim.dat", sep='\t')

reliable.groupby(['category']).describe() #.mean()

# for cat in set(reliable['category']):
#     #plt.figure()
#     #plt.title(cat)
#     cat_df = reliable.loc[reliable['category'] == cat] 
#     plt.plot(cat_df['words'], cat_df['days'], '.')
# plt.show()

X_train = reliable1_dim[:-20]
y_train = reliable['perday'][:-20]

#scaler = preprocessing.StandardScaler().fit(X_train)
#X_train_s = scaler.transform(X_train)

X_test = reliable1_dim[-20:]
y_test = reliable['perday'][-20:]

#X_test_s = scaler.transform(X_test)

regr = LinearRegression()
regr.fit(X_train, y_train)

y_pred = regr.predict(X_test)

print("Coefficients: \n{}".format(regr.coef_))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Variance score: %.3f" % r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.plot(range(400,1800), range(400,1800), 'k-')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.show()