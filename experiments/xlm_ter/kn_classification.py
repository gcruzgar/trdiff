import pandas as pd 
import torch
import numpy as np 

from scripts.utils import load_embeddings, remove_outliers

from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

ter = pd.read_csv("data/en-fr-100-mt_score.txt", sep='\n', header=None)
ter.columns=['score']

xlm_path = "data/xlm-embeddings/"
features = load_embeddings(xlm_path)

# Join data into single dataframe
df = ter.merge(features, left_on=ter.index, right_on=features.index)

def sk_classification(df, rm_out=False):

    # Remove outliers
    if rm_out == True:
        df = remove_outliers(df, 'score', lq=0.05, uq=0.95) 
        print("data points below 0.05 or above 0.95 quantiles removed")

    # Classify scores depedning on percentile
    df["class"] = 1 # average translation
    df.loc[df["score"] >= df["score"].quantile(0.67), "class"] = 0 # good translation
    df.loc[df["score"] <= df["score"].quantile(0.33), "class"] = 2 # bad translation

    # Split data into training and tests sets, set random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["score", "class"]), df["class"], test_size=0.2, random_state=42)

    print("running k-neighbors classifier...")

    results_dict = {}
    for n in range(3, 32):
        
        # Create classifier
        neigh = KNeighborsClassifier(n_neighbors=n, algorithm='auto')

        # Fit classifier to train data
        neigh.fit(X_train, y_train)

        results_dict[n] = neigh.score(X_test, y_test)
    
    results_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['kn-score'])
    max_score = max(results_df['kn-score'])

    print("maximum score obtained: %0.2f%%" % (max_score*100))

    max_list = results_df.loc[results_df['kn-score'] == max_score]

    for n in max_list.index: 
        
        # Create classifier
        neigh = KNeighborsClassifier(n_neighbors=n, algorithm='auto')

        # Fit classifier to train data
        neigh.fit(X_train, y_train)

        print("\nnumber of neighbours: %d" % n)
        
        # Predict using test data
        y_pred = neigh.predict(X_test)
        y_pred_prob = pd.DataFrame(neigh.predict_proba(X_test)).round(2)
        y_pred_prob.columns = ["prob 0", "prob 1", "prob 2"]

        # Evaluate results
        diff = {"good translation": 0, "average translation": 1, "bad translation": 2}

        y_res = pd.DataFrame(y_pred, columns=['y_pred'])
        y_res['y_test'] = y_test.values

        
        for key in diff.keys():
            
            key_val = y_res.loc[y_res["y_pred"] == diff[key]]
            print( "Accuracy for %s: %0.2f%%" % ( key, accuracy_score( key_val["y_test"], key_val["y_pred"] ) * 100 ) )

sk_classification(df, rm_out=False)