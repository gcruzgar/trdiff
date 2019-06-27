import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.utils import remove_outliers

wto = pd.read_csv("data/wto_timed_all.csv")

wto_french = remove_outliers(wto, 'PERDAY FRENCH', lq=0.05, uq=0.94).drop(columns=['DAYS SPANISH', 'PERDAY SPANISH'])
wto_spanish = remove_outliers(wto, 'PERDAY SPANISH', lq=0.05, uq=0.94).drop(columns=['DAYS FRENCH', 'PERDAY FRENCH'])

un = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
un = remove_outliers(un, 'perday', lq=0.05, uq=0.94)

def histogram():

    fig, axs = plt.subplots(1, 3, sharey=True)

    axs[0].hist(un['perday'], bins=20, edgecolor='black')
    axs[0].set_xlabel("Translation rate (words per day)")
    axs[0].set_ylabel("Frequency (documents)")
    axs[0].set_title("UN timed")

    axs[1].hist(wto_french['PERDAY FRENCH'], bins=25, edgecolor='black')
    axs[1].set_xlabel("Translation rate (words per day)")
    axs[1].set_title("WTO timed - French Translations")

    axs[2].hist(wto_spanish['PERDAY SPANISH'], bins=25, edgecolor='black')
    axs[2].set_xlabel("Translation rate (words per day)")
    axs[2].set_title("WTO timed - Spanish Translations")

    plt.show()

def combined_plot():

    fig, axs = plt.subplots(1, 3, sharey=True)

    sns.distplot(un['perday'], hist=True, kde=True, bins=20, hist_kws={'edgecolor':'black'}, kde_kws={'bw':200}, ax=axs[0])
    axs[0].set_xlabel("Translation rate (words per day)")
    axs[0].set_ylabel("Density")
    axs[0].set_title("UN Corpus - Spanish Translations")

    sns.distplot(wto_french['PERDAY FRENCH'], hist=True, kde=True, bins=25, hist_kws={'edgecolor':'black'}, kde_kws={'bw':200}, ax=axs[1])
    axs[1].set_xlabel("Translation rate (words per day)")
    axs[1].set_title("WTO - French Translations")

    sns.distplot(wto_spanish['PERDAY SPANISH'], hist=True, kde=True, bins=25, hist_kws={'edgecolor':'black'}, kde_kws={'bw':200}, ax=axs[2])
    axs[2].set_xlabel("Translation rate (words per day)")
    axs[2].set_title("WTO - Spanish Translations")

    plt.show()

wto.describe()
un.describe() 

#histogram()
combined_plot()