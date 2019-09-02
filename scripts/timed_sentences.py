import numpy as np
import pandas as pd 

from scripts.utils import remove_outliers, linear_regression

import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch

lan = "es"
if lan == "es":
    language = "Spanish"
elif lan == "fr":
    language = "French"

# Load time taken to translate and calculate sentence length
wpd = pd.read_csv("data/un-timed-sentences/en-"+lan+".processed", sep='\t').drop_duplicates()

words=[]
for i in wpd.index:
    words.append(len(wpd['Segment'][i].split()))
wpd["words"] = words 

# Filter empty sentences (with only serial number)
time = wpd.loc[~wpd['Segment'].str.contains("^\s*\S*[0-9]\S*\s*$"), ['Time-to-edit', 'words']].reset_index(drop=True)
time.columns= ["time (ms)", "words"]

""" TER - words per day"""
# Load TER scores
ter = pd.read_csv("data/un-timed-sentences/en-"+lan+"-gs.score", header=None, sep='\t')
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

# scatter plots
plt.scatter(df['perms'], df['score'])
plt.xlabel("words translated per ms")
plt.ylabel("TER")
plt.xlim([min(df['perms'])-0.0001, max(df['perms'])+0.0001])
#plt.scatter(q3['perms'], q3['score'])
plt.show()

plt.figure()
plt.scatter(dfr['words'], dfr['score'])
#plt.plot(np.unique(dfr['words']), np.poly1d(np.polyfit(dfr['words'], dfr['score'], 1))(np.unique(dfr['words'])), 'r--')
plt.xlabel("Sentence length (words)")
plt.ylabel("TER")
plt.title("Timed Sentences - %s" % language)
plt.show()

plt.figure()
plt.scatter(dfr['words'], dfr['time (ms)'])
plt.plot(np.unique(dfr['words']), np.poly1d(np.polyfit(dfr['words'], dfr['time (ms)'], 1))(np.unique(dfr['words'])), 'k--')
plt.xlabel("Sentence length (words)")
plt.ylabel("Time taken to translate (ms)")
plt.title("Timed Sentences - %s" % language)
#plt.show()

# Line of best fit and distance from each point to the line
c, m = np.polynomial.polynomial.polyfit(dfr['words'], dfr['time (ms)'], 1)
y_pred = m*dfr['words'] + c 
residuals = dfr['time (ms)'] - y_pred
median_error = abs(residuals).median()
MAE = mean_absolute_error(dfr['time (ms)'], y_pred) # mean absolute error
plt.plot(dfr['words'], y_pred+MAE, 'r--') # confidence intervals (bestfit +/- MAE)
plt.plot(dfr['words'], y_pred-MAE, 'r--')
pos_res = residuals.loc[residuals >  MAE] # points above the line
neg_res = residuals.loc[residuals < -MAE] # points below the line
plt.show()

# Load biber dimension and select useful dimensions
biber = pd.read_csv("data/un-timed-sentences/en-"+lan+"-biber.en", sep='\t')
drop_cols = biber.columns[(biber == 0).sum() > 0.5*biber.shape[0]]
biber.drop(drop_cols, axis=1, inplace=True)

pos_res_df = biber.loc[pos_res.index]
neg_res_df = biber.loc[neg_res.index]

# biber plots
fig, axs = plt.subplots(1, 3, figsize=(15,15))
fig.suptitle('Timed Sentences - %s' % language, fontsize=16)

axs[0].scatter(pos_res_df['f26.NOUN'], pos_res_df['f27.wordLength'])
axs[0].scatter(neg_res_df['f26.NOUN'], neg_res_df['f27.wordLength'])
axs[0].set_xlabel('f26.NOUN')
axs[0].set_ylabel('f27.wordLength')
axs[0].legend(['below-lobf', 'above-lobf'])

axs[1].scatter(pos_res_df['f28.ADP'], pos_res_df['f29.typeTokenRatio'])
axs[1].scatter(neg_res_df['f28.ADP'], neg_res_df['f29.typeTokenRatio'])
axs[1].set_xlabel('f28.ADP')
axs[1].set_ylabel('f29.typeTokenRatio')
axs[1].legend(['below-lobf', 'above-lobf'])

if lan=='fr':
    axs[2].scatter(pos_res_df['f30.attributiveAdjectives'], pos_res_df['f57.conjuncts'])
    axs[2].scatter(neg_res_df['f30.attributiveAdjectives'], neg_res_df['f57.conjuncts'])
    axs[2].set_xlabel('f30.attributiveAdjectives')
    axs[2].set_ylabel('f57.conjuncts')
    axs[2].legend(['below-lobf', 'above-lobf'])
elif lan=='es':
    axs[2].scatter(pos_res_df['f30.attributiveAdjectives'], pos_res_df['f47.nominalizations'])
    axs[2].scatter(neg_res_df['f30.attributiveAdjectives'], neg_res_df['f47.nominalizations'])
    axs[2].set_xlabel('f30.attributiveAdjectives')
    axs[2].set_ylabel('f47.nominalizations')
    axs[2].legend(['below-lobf', 'above-lobf'])

plt.show()

""" XLM - words per day / TER """
def xlm_regression(var_name='perms'):

    # Load sentece embeddings
    features = pd.DataFrame(torch.load("data/un-timed-sentences/en-"+lan+"-gs-xlm-embeddings.pt").data.numpy())
    reg_df = dfr.merge(features, left_index=True, right_index=True)

    print("Predicting %s..." % var_name)
    ols, scaler, X_test, y_test = linear_regression(reg_df.drop(columns=["score", "time (ms)", "words", "perms", "rate"]), reg_df[var_name])

def xlm_classification(var_name='perms'):
    
    # Load sentece embeddings
    features = pd.DataFrame(torch.load("data/un-timed-sentences/en-"+lan+"-gs-xlm-embeddings.pt").data.numpy())

    # Classify objective variable based on percentile
    dfr["class"] = 1 # average
    dfr.loc[df[var_name] >= dfr[var_name].quantile(0.67), "class"] = 0 # above average
    dfr.loc[df[var_name] <= dfr[var_name].quantile(0.33), "class"] = 2 # below average

    # Split data into training and tests sets, set random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(features.loc[dfr.index], dfr["class"], test_size=0.2, random_state=42)

    # Create classifier
    C=1
    clf = SVC(C=C, kernel='rbf', gamma='scale')

    # Fit classifier to train data
    clf.fit(X_train, y_train)

    # Predict and evaluate results
    y_pred = clf.predict(X_test)
    diff = {"above average": 0, "average": 1, "below average": 2}
    print("\nClassification report (%s):\n" % var_name)
    print(classification_report(y_test, y_pred, target_names=diff))

#xlm_regression(var_name='score')
#xlm_classification(var_name='score')

""" kde plots """
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
plt.title("Timed Sentence Translation - %s" % language)

plt.show()