import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt

""" Correlation between XLM embeddings and TER using scikit-learn """
from scripts.utils import linear_regression

# Load TER scores (only first 100 due to memory limit)
mt_scores = pd.read_csv("data/en-fr-100-mt_score.txt", sep='\n', header=None)[0:100]
mt_scores.columns=['score']

# Load XLM embeddings
features = pd.read_csv("data/en-fr-embeddings_top100.csv")

df = pd.concat([mt_scores, features], axis=1)
ols, scaler, X_test, y_test = linear_regression(df.drop(columns=['score']), df['score'].values, test_size=0.15)[0:2]

""" Correlation with TER using pytorch """

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
