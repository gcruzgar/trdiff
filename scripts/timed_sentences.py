import numpy as np
import pandas as pd 

from scripts.utils import remove_outliers, linear_regression

import seaborn as sns 
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import normalize
#from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch

lan = "all"
if lan == "es":
    language = "Spanish"
elif lan == "fr":
    language = "French"
elif lan == "all":
    language = "French & Spanish"

# Load time taken to translate and calculate sentence length
if lan == "all":
    es = pd.read_csv("data/un-timed-sentences/en-es.processed", sep='\t')
    fr = pd.read_csv("data/un-timed-sentences/en-fr.processed", sep='\t')
    wpd = pd.concat([es,fr], axis=0, ignore_index=True).drop_duplicates()
else:
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
if lan == "all":
    es = pd.read_csv("data/un-timed-sentences/en-es-gs.score", header=None, sep='\t')
    fr = pd.read_csv("data/un-timed-sentences/en-fr-gs.score", header=None, sep='\t')
    ter = pd.concat([es,fr], axis=0, ignore_index=True)
else:
    ter = pd.read_csv("data/un-timed-sentences/en-"+lan+"-gs.score", header=None, sep='\t')
ter.columns = ["score"]

# Join important columns to single dataframe
df = pd.concat([ter, time], axis=1)

# Calculate translation rate (and normalise)
#df['perms'] = df['words'] / df['time (ms)'] # words per ms
df['spw'] =  (df['time (ms)'])/1000 / df['words'] # seconds per word
#df['rate'] = (df['perms'] - df['perms'].min()) / (df['perms'].max() - df['perms'].min())

# Remove perfect translations
dft = df.loc[df['score'] != 0]

# Remove outliers
dfr = remove_outliers(df, 'spw', lq=0.05, uq=0.95)

# Correlation
print(dfr.corr().round(3)['score'])

# Quantiles
def quantiles(df):
    """ Output distribution of each quantile in the data set. """

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

#dfr = dfr.loc[dfr['spw'] < 8] # filter out extreme cases
dfr = dfr.loc[dfr['score'] <= 1.0] 
dfr = dfr.loc[dfr['words'] <= 90]

# scatter plots
plt.scatter(dfr['spw'], dfr['score'])
plt.xlabel("seconds per word")
plt.ylabel("TER")
#plt.xlim([min(df['spw'])-0.0001, max(df['spw'])+0.0001])
#plt.scatter(q3['perms'], q3['score'])

c, m = np.polynomial.polynomial.polyfit(dfr['spw'], dfr['score'], 1)
y_pred = m*dfr['spw'] + c 
residuals = dfr['score'] - y_pred
median_error = abs(residuals).median()
MAE = mean_absolute_error(dfr['score'], y_pred) # mean absolute error

plt.plot(np.unique(dfr['spw']), np.poly1d(np.polyfit(dfr['spw'], dfr['score'], 1))(np.unique(dfr['spw'])), 'k--')

x1 = np.linspace(min(dfr['spw']), max(dfr['spw']))
y1 = m*x1 + c

plt.plot(x1, y1+MAE, 'r--') # confidence intervals (bestfit +/- MAE)
plt.plot(x1, y1-MAE, 'r--')

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

x1 = np.linspace(min(dfr['words']), max(dfr['words']))
y1 = m*x1 + c

plt.plot(x1, y1+MAE, 'r--') # confidence intervals (bestfit +/- MAE)
plt.plot(x1, y1-MAE, 'r--')

plt.show()

pos_res = residuals.loc[residuals >  MAE] # points above the line
neg_res = residuals.loc[residuals < -MAE] # points below the line

# Load biber dimension and select useful dimensions
if lan == "all":
    es = pd.read_csv("data/un-timed-sentences/en-es-biber.en", sep='\t')
    fr = pd.read_csv("data/un-timed-sentences/en-fr-biber.en", sep='\t')
    biber = pd.concat([es,fr], axis=0, ignore_index=True)
else:
    biber = pd.read_csv("data/un-timed-sentences/en-"+lan+"-biber.en", sep='\t')
drop_cols = biber.columns[(biber == 0).sum() > 0.75*biber.shape[0]]
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

""" XLM / BERT - words per day / TER """
def embedding_regression(var_name='spw', model='xlm'):

    # Load sentece embeddings
    if model == 'xlm' or model == 'XLM':
        features = pd.DataFrame(torch.load("data/un-timed-sentences/en-"+lan+"-gs-xlm-embeddings.pt").data.numpy())
    elif model == 'bert' or model == 'BERT':
        features = pd.read_csv("data/un-timed-sentences/bert-embeddings-timed-sentences-"+lan+"-multi.csv", header=None)

    reg_df = dfr.merge(features, left_index=True, right_index=True)

    print("Predicting %s using %s..." % (var_name, model))
    ols, scaler, X_test, y_test = linear_regression(reg_df.drop(columns=["score", "time (ms)", "words", "spw"]), reg_df[var_name])

def embedding_classification(var_name='spw', model='xlm'):

    # Load sentece embeddings
    if model == 'xlm' or model == 'XLM':
        features = pd.DataFrame(torch.load("data/un-timed-sentences/en-"+lan+"-gs-xlm-embeddings.pt").data.numpy())
    elif model == 'bert' or model == 'BERT':
        features = pd.read_csv("data/un-timed-sentences/bert-embeddings-timed-sentences-"+lan+"-multi.csv", header=None)
        #scaler = MinMaxScaler()
        #features = scaler.fit_transform(features)
        features = normalize(features, axis=0)

    # Classify objective variable based on percentile
    dfr["class"] = 1 # average
    dfr.loc[df[var_name] >= dfr[var_name].quantile(0.67), "class"] = 0 # above average
    dfr.loc[df[var_name] <= dfr[var_name].quantile(0.33), "class"] = 2 # below average

    # Split data into training and tests sets, set random_state for reproducibility

    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(features).loc[dfr.index], dfr["class"], test_size=0.2, random_state=42)

    # Create classifier
    C=1
    clf = SVC(C=C, kernel='rbf', gamma='scale')

    # Fit classifier to train data
    clf.fit(X_train, y_train)

    # Predict and evaluate results
    y_pred = clf.predict(X_test)
    diff = {"above average": 0, "average": 1, "below average": 2}
    print("\nClassification report (%s - %s):\n" % (var_name, model))
    print(classification_report(y_test, y_pred, target_names=diff))

#embedding_regression(var_name='score', model='bert')
#embedding_classification(var_name='score', model='bert')

from sklearn.linear_model import Ridge
def ridge_regression(var_name='spw', model='xlm'):
    
    # Load sentece embeddings
    if model == 'xlm' or model == 'XLM':
        features = pd.DataFrame(torch.load("data/un-timed-sentences/en-"+lan+"-gs-xlm-embeddings.pt").data.numpy())
    elif model == 'bert' or model == 'BERT':
        features = pd.read_csv("data/un-timed-sentences/bert-embeddings-timed-sentences-"+lan+".csv", header=None)

    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(features).loc[dfr.index], dfr[var_name], test_size=0.2)

    rreg = Ridge(alpha=10)
    rreg.fit(X_train, y_train)
    y_pred = rreg.predict(X_test)
    
    # Real vs Predicted
    plt.figure()
    plt.scatter(y_test, y_pred)
    #plt.plot(range(0, 1), range(0, 1), 'k-')
    plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, y_pred, 1))(np.unique(y_test)), 'r--')
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('OLS - Real vs Predicted')
    plt.show()
    print("r2-score: %.3f" % rreg.score(X_test, y_test))

#ridge_regression(var_name='score', model='bert')

""" kde plots """
pl = "spw"
if pl=="spw":
    sns.distplot(dfr['spw']*100, hist=True, kde=True, bins=20, hist_kws={'edgecolor': 'black'}, kde_kws={'bw': 50})
    plt.xlabel("Time spent translating (seconds per word)")
elif pl=="ter":
    sns.distplot(dfr['score'], hist=True, kde=True, bins=20, hist_kws={'edgecolor': 'black'}, kde_kws={'bw': 0.05})
    plt.xlabel("TER")
elif pl=="words":
    sns.distplot(dfr['words'], hist=True, kde=True, bins=20, hist_kws={'edgecolor': 'black'}, kde_kws={'bw': 5})
    plt.xlabel("Sentence Length (number of words)")

plt.ylabel("Density")
plt.title("Timed Sentence Translation - %s" % language)

plt.show()

""" predicting sentences """
from sklearn.svm import SVR 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, classification_report

if lan == "all":
    es = pd.DataFrame(torch.load("data/un-timed-sentences/en-es-gs-xlm-embeddings.pt").data.numpy())
    fr = pd.DataFrame(torch.load("data/un-timed-sentences/en-fr-gs-xlm-embeddings.pt").data.numpy())
    features = pd.concat([es,fr], axis=0, ignore_index=True)
else:
    features = pd.DataFrame(torch.load("data/un-timed-sentences/en-"+lan+"-gs-xlm-embeddings.pt").data.numpy())

spw = dfr['spw']

#X_test = torch.load("data/xlm-predict_sentences.pt").data.numpy()

reg_df = features.merge(spw, left_index=True, right_index=True)

def sentence_regression(reg_df, method='lr'):

    if method == 'lr':
        model = LinearRegression()
    elif method == 'svr':
        model = SVR(kernel='linear', C=1.0)
    if method == 'abr':
        model = AdaBoostRegressor(n_estimators=50)

    X = reg_df.drop(columns=['spw'])
    y = reg_df['spw']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))

    y_pred = model.predict(X_test)
    print(mean_absolute_error(y_test, y_pred))

sentence_regression(reg_df, method='lr')

# Classify objective variable based on percentile
reg_df["class"] = 1 # average
reg_df.loc[reg_df['spw'] >= reg_df['spw'].quantile(0.67), "class"] = 0 # above average
reg_df.loc[reg_df['spw'] <= reg_df['spw'].quantile(0.33), "class"] = 2 # below average

def sentence_classification(method='svc'): 

    if method == 'svc':
        model = SVC(gamma='auto', kernel='linear', C=10.0)
    elif method == 'abc':
        model = AdaBoostClassifier(algorithm='SAMME.R', n_estimators=50)
    elif method == 'mlp':
        model = MLPClassifier(activation="relu", solver="adam", alpha=0.1, random_state=42)

    X = reg_df.drop(columns=['spw', "class"])
    y = reg_df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    diff = {"long time": 0, "average time": 1, "short time": 2}
    print(classification_report(y_test, y_pred, target_names=diff))

""" cross-validation """
from scripts.utils import kfold_crossvalidation
X = reg_df.drop(columns=['spw','class'])
y = reg_df['spw'] # 'class' if classification
y_df = kfold_crossvalidation(X, y, k=10, method='reg', output='df')

#method = 'clf'
#diff = {"slow translation": 0, "average translation": 1, "fast translation": 2}
#print(classification_report(y_df['y_test'], y_df['y_pred'], target_names=diff))

#method = 'reg'
print("Correlation = %0.3f" % y_df.corr().iloc[0,1])
print("MAE = %0.3f" %mean_absolute_error(y_df['y_test'], y_df['y_pred']))