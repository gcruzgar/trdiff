#!/usr/bin/env python3

import pandas as pd 
import numpy as np 

from utils import remove_outliers

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

""" Linear Regression using scikit-learn """
def sklearn_regression():    
    from scripts.utils import linear_regression

    # Load TER scores (only first 100 due to memory limit)
    mt_scores = pd.read_csv("data/en-fr-100-mt_score.txt", sep='\n', header=None)[0:100]
    mt_scores.columns=['score']

    # Load XLM embeddings
    features = pd.read_csv("data/en-fr-embeddings_top100.csv")

    df = pd.concat([mt_scores, features], axis=1)
    ols, scaler, X_test, y_test = linear_regression(df.drop(columns=['score']), df['score'].values, test_size=0.15)[0:2]

""" Linear Regression using pytorch """
def pytorch_regression():
    # Load TER scores (only first 100 due to memory limit)
    mt_scores = pd.read_csv("en-fr-100/en-fr-100-mt_score.txt", sep='\n', header=None)[0:100] 
    mt_scores.columns=['score']

    # features and target
    X = tensor[0]
    y = torch.from_numpy(mt_scores.values).float()

    # split data into train and test sets
    X_train = X[:-15]
    X_test = X[-15:]
    y_train = y[:-15]
    y_test = y[-15:]

    # define linear regression
    class LinearRegression(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super(LinearRegression, self).__init__()
            self.linear = torch.nn.Linear(input_size, output_size)
        def forward(self, x):
            out = self.linear(x)
            return out

    input_size = X.shape[1]
    output_size = 1
    num_epochs = 50
    learning_rate = 0.001

    reg = LinearRegression(input_size, output_size)

    # loss and optimization criteria
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(reg.parameters(), lr=learning_rate)

    from torch.autograd import Variable

    # train model
    for epoch in range(num_epochs):
        inputs = Variable(X_train)
        labels = Variable(y_train)
        optimizer.zero_grad() # don't want gradient from previous epoch to carry over
        # get outputs from model
        outputs = reg(inputs)
        # get loss for predicted output
        loss = criterion(outputs, labels)
        loss.backward()
        # update parameters
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    # test model
    with torch.no_grad():
        y_pred = reg(Variable(X_test))

    plt.plot(y_test.data.numpy(), y_pred.data.numpy(), '.')
    plt.show()


""" Classification """
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier

def sk_classification():
    # Load TER scores (will only use first 100 due to memory limit on tensor)
    mt_scores = pd.read_csv("data/en-fr-100-mt_score.txt", sep='\n', header=None)
    mt_scores.columns=['score']

    # Load XLM embeddings
    features = pd.read_csv("data/en-fr-embeddings_top100.csv")

    # Join data into single dataframe
    df = pd.concat([mt_scores[0:100], features], axis=1)

    # Remove outliers
    df = remove_outliers(df, 'score', lq=0.05, uq=0.95) 

    # Classify scores depedning on percentile
    df["class"] = 1 # average translation
    df.loc[df["score"] >= df["score"].quantile(0.67), "class"] = 0 # good translation
    df.loc[df["score"] <= df["score"].quantile(0.33), "class"] = 2 # bad translation

    # df["class"] = 1 # good translation
    # df.loc[df["score"] > df["score"].quantile(0.75), "class"] = 0 # very good translation
    # df.loc[df["score"] <= df["score"].quantile(0.25), "class"] = 3 # very bad translation
    # df.loc[(df["score"] > df["score"].quantile(0.25)) & (df["score"] <= df["score"].quantile(0.5)), "class"] = 2 # bad translation

    #print("Number of documents per class: \n{}".format(df["class"].value_counts(sort=False).to_string()))

    # Split data into training and tests sets, set random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["score", "class"]), df["class"], test_size=0.2, random_state=42)

    # Create classifier
    neigh = KNeighborsClassifier(n_neighbors=4, algorithm='auto')

    # Fit classifier to train data
    neigh.fit(X_train, y_train)

    # Predict using test data
    y_pred = neigh.predict(X_test)
    y_pred_prob = pd.DataFrame(neigh.predict_proba(X_test)).round(2)
    y_pred_prob.columns = ["prob 0", "prob 1", "prob 2"]
    #y_pred_prob.columns = ["prob 0", "prob 1", "prob 2", "prob 3"]

    # Evaluate results
    diff = {"good translation": 0, "average translation": 1, "bad translation": 2}
    #diff = {"very good translation": 0, "good translation": 1, "bad translation": 2, "very bad translation": 3}

    y_res = pd.DataFrame(y_pred, columns=['y_pred'])
    y_res['y_test'] = y_test.values

    for key in diff.keys():
        
        key_val = y_res.loc[y_res["y_pred"] == diff[key]]
        print( "Accuracy for %s: %0.2f%%" % ( key, accuracy_score( key_val["y_test"], key_val["y_pred"] ) * 100 ) )

#sklearn_regression()
#pytorch_regression()
sk_classification()