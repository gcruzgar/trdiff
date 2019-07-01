import numpy as np 
import pandas as pd 

import torch
from scripts.utils import load_embeddings, remove_outliers

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ter = pd.read_csv("data/en-fr-100-mt_score.txt", sep='\n', header=None)
ter.columns=['score']

xlm_path = "data/xlm-embeddings/"
features = load_embeddings(xlm_path)

# Use non-zero biber dimensions as features
use_biber=False
if use_biber == True:
    biber = pd.read_csv("data/en-fr-100.dim", sep='\t')

    drop_cols = biber.columns[(biber == 0).sum() > 0.5*biber.shape[0]]
    biber.drop(drop_cols, axis=1, inplace=True)

    features = features.merge(biber, left_index=True, right_index=True)

# Join data into single dataframe
df = ter.merge(features, left_index=True, right_index=True)

# Remove outliers
rm_out=False
if rm_out == True:
    df = remove_outliers(df, 'score', lq=0.05, uq=0.95) 
    print("data points below 0.05 or above 0.95 quantiles removed")

# Classify scores based on percentile
df["class"] = 1 # average translation
df.loc[df["score"] >= df["score"].quantile(0.67), "class"] = 0 # good translation
df.loc[df["score"] <= df["score"].quantile(0.33), "class"] = 2 # bad translation

# Split data into training and tests sets, set random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["score", "class"]), df["class"], test_size=0.2, random_state=42)

# for a in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]:

#     # Create classifier
#     mlp = MLPClassifier(activation="relu", solver="adam", alpha=a, random_state=42)

#     # Fit classifier to train data
#     mlp.fit(X_train, y_train)

#     # Predictions
#     y_pred = mlp.predict(X_test)

#     # Evaluate results
#     print("alpha = %0.5f, score = %0.3f" % (a, mlp.score(X_test, y_test)))

# Create classifier
mlp = MLPClassifier(activation="relu", solver="adam", alpha=0.1, random_state=42)

# Fit classifier to train data
mlp.fit(X_train, y_train)

# Predict and evaluate results
print("\nclassification report:\n")
y_pred = clf.predict(X_test)
diff = {"good translation": 0, "average translation": 1, "bad translation": 2}
print(classification_report(y_res['y_test'], y_res['y_pred'], target_names=diff))