import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.utils import remove_outliers

wto = pd.read_csv("data/wto_timed_all.csv")

wto_french = remove_outliers(wto, 'PERDAY FRENCH', lq=0.05, uq=0.94).drop(columns=['DAYS SPANISH', 'PERDAY SPANISH'])
wto_spanish = remove_outliers(wto, 'PERDAY SPANISH', lq=0.05, uq=0.94).drop(columns=['DAYS FRENCH', 'PERDAY FRENCH'])

un = pd.read_csv("data/timed-un/reliable.dat", sep=' ')
#un = remove_outliers(un, 'perday', lq=0.05, uq=0.94)

def combined_plot():

    fig, axs = plt.subplots(1, 3, sharey=True)

    sns.distplot(un['perday'], hist=True, kde=True, bins=20, hist_kws={'edgecolor':'black'}, kde_kws={'bw':200}, ax=axs[0])
    axs[0].set_xlabel("Translation rate (words per day)")
    axs[0].set_ylabel("Density")
    axs[0].set_title("UN Corpus")

    sns.distplot(wto_french['PERDAY FRENCH'], hist=True, kde=True, bins=25, hist_kws={'edgecolor':'black'}, kde_kws={'bw':200}, ax=axs[1])
    axs[1].set_xlabel("Translation rate (words per day)")
    axs[1].set_title("WTO - French Translations")

    sns.distplot(wto_spanish['PERDAY SPANISH'], hist=True, kde=True, bins=25, hist_kws={'edgecolor':'black'}, kde_kws={'bw':200}, ax=axs[2])
    axs[2].set_xlabel("Translation rate (words per day)")
    axs[2].set_title("WTO - Spanish Translations")

    plt.show()

def compare_un_wto_averages():
    # Average average translation rate
    french = wto['PERDAY FRENCH']
    spanish = wto['PERDAY SPANISH']
    wto_ave = (french+spanish)/2

    # Remove outliers
    iqr = wto_ave.quantile(q=0.75) - wto_ave.quantile(q=0.25)
    wto_ave = wto_ave[wto_ave > wto_ave.quantile(q=0.5) - 1.5*iqr]
    wto_ave = wto_ave[wto_ave < wto_ave.quantile(q=0.5) + 1.5*iqr]

    fig, axs = plt.subplots(1, 2, sharey=True)

    sns.distplot(un['perday'], hist=True, kde=True, bins=20, hist_kws={'edgecolor':'black'}, kde_kws={'bw':200}, ax=axs[0])
    axs[0].set_xlabel("Translation rate (words per day)")
    axs[0].set_ylabel("Density")
    axs[0].set_title("UN Corpus - Spanish Translations")

    sns.distplot(wto_ave, hist=True, kde=True, bins=25, hist_kws={'edgecolor':'black'}, kde_kws={'bw':100}, ax=axs[1])
    axs[1].set_xlabel("Translation rate (words per day)")
    axs[1].set_title("WTO - Average")

    plt.show()

wto.describe()
un.describe() 
combined_plot()
compare_un_wto_averages()