import pandas as pd 
import numpy as np 
import torch

from scripts.utils import load_embeddings, remove_outliers, linear_regression


ter = pd.read_csv("data/en-fr-100-mt_score.txt", sep='\n', header=None)
ter.columns=['score']

xlm_path = "data/xlm-embeddings/"
features = load_embeddings(xlm_path)

# Join data into single dataframe
df = ter.merge(features, left_on=ter.index, right_on=features.index)

# Remove outliers
rm_out=True
if rm_out == True:
    df = remove_outliers(df, 'score', lq=0.05, uq=0.95) 
    print("data points below 0.05 or above 0.95 quantiles removed")
    
ols, scaler, X_test, y_test = linear_regression(df.drop(columns=['score']), df['score'])