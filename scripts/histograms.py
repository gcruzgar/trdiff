import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.utils import remove_outliers

es_df = pd.read_csv("data/es-mt-score.txt", header=None, sep='\n')
es_df.columns=['score']

es_df = remove_outliers(es_df, 'score', lq=0.05, uq=0.95)

fr_df = pd.read_csv("data/fr-mt-score.txt", header=None, sep='\n')
fr_df.columns=['score']

fr_df = remove_outliers(fr_df, 'score', lq=0.05, uq=0.95)

def histogram():

    fig, axs = plt.subplots(1, 2, sharey=True)

    axs[0].hist(es_df.iloc[:,0], bins=25, edgecolor='black')
    axs[0].set_xlabel("TER")
    axs[0].set_ylabel("Frequency (sentences)")
    axs[0].set_title("UN Corpus - Spanish Translations")

    axs[1].hist(fr_df.iloc[:,0], bins=25, edgecolor='black')
    axs[1].set_xlabel("TER")
    #axs[1].ylabel("Frequency (sentences)")
    axs[1].set_title("UN Corpus - French Translations")

    plt.show()

def KernelDensity():

    fig, axs = plt.subplots(1, 2, sharey=True)

    sns.kdeplot(es_df.iloc[:,0], bw=0.05, ax=axs[0], legend=False)
    axs[0].set_xlabel("TER")
    axs[0].set_ylabel("Density")
    axs[0].set_title("UN Corpus - Spanish Translations")

    sns.kdeplot(fr_df.iloc[:,0], bw=0.05, ax=axs[1], legend=False)
    axs[1].set_xlabel("TER")
    #axs[1].ylabel("Frequency (sentences)")
    axs[1].set_title("UN Corpus - French Translations")

    plt.show()

def combined_plot():

    fig, axs = plt.subplots(1, 2, sharey=True)

    sns.distplot(es_df.iloc[:,0], hist=True, kde=True, bins=25, hist_kws={'edgecolor':'black'}, kde_kws={'bw':0.05}, ax=axs[0])#, label=False)
    axs[0].set_xlabel("TER")
    axs[0].set_ylabel("Density")
    axs[0].set_title("UN Corpus - Spanish Translations")

    sns.distplot(fr_df.iloc[:,0], hist=True, kde=True, bins=25, hist_kws={'edgecolor':'black'}, kde_kws={'bw':0.05}, ax=axs[1])#, label=False)
    axs[1].set_xlabel("TER")
    #axs[1].ylabel("Frequency (sentences)")
    axs[1].set_title("UN Corpus - French Translations")

    plt.show() 

#histogram()
#KernelDensity()
combined_plot()