import numpy as np
import pandas as pd 

from scripts.utils import remove_outliers, linear_regression

import torch

lan = "es"
# Load time taken to translate and calculate sentence length
wpd = pd.read_csv("data/golden-standard/en-"+lan+".pe", sep='\t').drop_duplicates()

words=[]
for i in wpd.index:
    words.append(len(wpd['Segment'][i].split()))
wpd["words"] = words 

# Filter empty sentences (with only serial number)
time = wpd.loc[~wpd['Segment'].str.contains("^\s*\S*[0-9]\S*\s*$"), ['Time-to-edit', 'words']].reset_index(drop=True)
time.columns= ["time (ms)", "words"]

""" TER - words per day"""
# Load TER scores
ter = pd.read_csv("data/golden-standard/en-"+lan+"-gs.score", header=None, sep='\t')
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
print(dfr.corr().round(3)['score'])

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
    features = pd.DataFrame(torch.load("data/golden-standard/en-"+lan+"-gs-xlm-embeddings.pt").data.numpy())
    reg_df = df.merge(features, left_index=True, right_index=True)

    ols, scaler, X_test, y_test = linear_regression(reg_df.drop(columns=["score", "time (ms)", "words", "perms", "rate"]), reg_df['perms'])

def xlm_classification():
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    
    # Load sentece embeddings
    features = pd.DataFrame(torch.load("data/golden-standard/en-"+lan+"-gs-xlm-embeddings.pt").data.numpy())

    # Classify translation rate based on percentile
    df["class"] = 1 # average rate
    df.loc[df["rate"] >= df["rate"].quantile(0.67), "class"] = 0 # fast rate
    df.loc[df["rate"] <= df["rate"].quantile(0.33), "class"] = 2 # slow rate

    # Split data into training and tests sets, set random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(features, df["class"], test_size=0.2, random_state=42)

    # Create classifier
    C=1
    clf = SVC(C=C, kernel='rbf', gamma='scale')

    # Fit classifier to train data
    clf.fit(X_train, y_train)

    # Predict and evaluate results
    y_pred = clf.predict(X_test)
    diff = {"fast rate": 0, "average rate": 1, "slow rate": 2}
    print("\nclassification report:\n")
    print(classification_report(y_test, y_pred, target_names=diff))

#xlm_regression()
xlm_classification()

""" kde plots """
import seaborn as sns 
pl = "rate"
if pl=="rate":
    sns.distplot(dfr['perms']*100, hist=True, kde=True, bins=15, hist_kws={'edgecolor': 'black'}, kde_kws={'bw': 0.015})
    plt.xlabel("Translation Rate (words per second)")
elif pl=="ter":
    sns.distplot(dfr['score'], hist=True, kde=True, bins=15, hist_kws={'edgecolor': 'black'}, kde_kws={'bw': 0.05})
    plt.xlabel("TER")
elif pl=="words":
    sns.distplot(dfr['words'], hist=True, kde=True, bins=15, hist_kws={'edgecolor': 'black'}, kde_kws={'bw': 5})
    plt.xlabel("Sentence Length (number of words)")

plt.ylabel("Density")

if lan == "es":
    plt.title("Timed Sentence Translation - Spanish")
elif lan == "fr":
    plt.title("Timed Sentence Translation - French")

plt.show()