import torch
import numpy as np
import pandas as pd 
from scripts.utils import remove_outliers, linear_regression

# Load time taken to translate and calculate sentence length
wpd = pd.read_csv("data/golden-standard/en-es.pe", sep='\t').drop_duplicates()
wpd['words'] = 0
for i in wpd.index:
    wpd['words'][i] = len(wpd['Segment'][i].split())

# Filter empty sentences (with only serial number)
time = wpd.loc[~wpd['Segment'].str.contains("^\s*\S*[0-9]\S*\s*$"), ['Time-to-edit', 'words']].reset_index(drop=True)
time.columns= ["time (ms)", "words"]

""" TER - words per day"""
# Load TER scores
ter = pd.read_csv("data/golden-standard/en-es-gs.score", header=None, sep='\t')
ter.columns = ["score"]

# Join important columns to single dataframe
df = pd.concat([ter, time], axis=1)

# Calculate translation rate (and normalise)
df['perms'] = df['words'] / df['time (ms)']
df['rate'] = (df['perms'] - df['perms'].min()) / (df['perms'].max() - df['perms'].min())

# Remove perfect translations
dft = df.loc[df['score'] != 0]

# Remove outliers
dfr = remove_outliers(df, 'rate', lq=0.05, uq=0.95)

# Correlation
print(dft.corr().round(3)['score'])

# Quantiles
q1 = df.loc[df['perms'] <= df['perms'].quantile(0.25)]
q2 = df.loc[(df['perms'] >= df['perms'].quantile(0.25)) & (df['perms'] <= df['perms'].quantile(0.50))]
q3 = df.loc[(df['perms'] >= df['perms'].quantile(0.50)) & (df['perms'] <= df['perms'].quantile(0.75))]
q4 = df.loc[df['perms'] >= df['perms'].quantile(0.75)]

q_corr={}
q_df={1:q1, 2:q2, 3:q3, 4:q4}
for q in range(1,5):
    q_corr[q] = q_df[q].corr()['score']

qcor_df = pd.DataFrame.from_dict(q_corr)
qcor_df.columns=['q1', 'q2', 'q3', 'q4']

print(qcor_df.round(3))

#plots
import matplotlib.pyplot as plt
plt.scatter(df['perms'], df['score'])
plt.xlabel("words translated per ms")
plt.ylabel("TER")
plt.xlim([min(df['perms'])-0.0001, max(df['perms'])+0.0001])
#plt.scatter(q3['perms'], q3['score'])
plt.show()

""" XLM - words per day """
def xlm_regression():
    # Load sentece embeddings
    features = pd.DataFrame(torch.load("data/golden-standard/en-es-gs-xlm-embeddings.pt").data.numpy())
    reg_df = df.merge(features, left_index=True, right_index=True)

    ols, scaler, X_test, y_test = linear_regression(reg_df.drop(columns=["score", "time (ms)", "words", "perms", "rate"]), reg_df['perms'])

#xlm_regression()