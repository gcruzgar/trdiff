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

# Remove outliers
df = remove_outliers(df, 'rate', lq=0.05, uq=0.95)

# Correlation
print(df.corr().round(3)['score'])

""" XLM - words per day """
def xlm_regression():
    # Load sentece embeddings
    features = pd.DataFrame(torch.load("data/golden-standard/en-es-gs-xlm-embeddings.pt").data.numpy())
    reg_df = df.merge(features, left_index=True, right_index=True)

    ols, scaler, X_test, y_test = linear_regression(reg_df.drop(columns=["score", "time (ms)", "words", "perms", "rate"]), reg_df['perms'])

#xlm_regression()