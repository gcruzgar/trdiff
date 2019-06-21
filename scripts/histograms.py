import pandas as pd
import matplotlib.pyplot as plt
from scripts.utils import remove_outliers

es_df = pd.read_csv("data/es-mt-score.txt", header=None, sep='\n')
es_df.columns=['score']

es_df = remove_outliers(es_df, 'score', lq=0.05, uq=0.95)

fr_df = pd.read_csv("data/fr-mt-score.txt", header=None, sep='\n')
fr_df.columns=['score']

fr_df = remove_outliers(fr_df, 'score', lq=0.05, uq=0.95)

fig, axs = plt.subplots(1, 2, sharey=True)

axs[0].hist(es_df.iloc[:,0], bins=40)
axs[0].set_xlabel("TER")
axs[0].set_ylabel("Frequency (sentences)")
axs[0].set_title("UN Corpus - Spanish Translations")

axs[1].hist(fr_df.iloc[:,0], bins=40)
axs[1].set_xlabel("TER")
#axs[1].ylabel("Frequency (sentences)")
axs[1].set_title("UN Corpus - French Translations")

plt.show()