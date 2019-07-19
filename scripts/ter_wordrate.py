import pandas as pd 
from scripts.utils import remove_outliers

# Load TER scores
ter = pd.read_csv("data/en-es-gs.score", header=None, sep='\t')
ter.columns = ["score"]

# Load time taken to translate and calculate sentence length
wpd = pd.read_csv("data/en-es.pe", sep='\t').drop_duplicates()
wpd['words'] = 0
for i in wpd.index:
    wpd['words'][i] = len(wpd['Segment'][i].split())

# Filter empty sentences (with only serial number)
time = wpd.loc[~wpd['Segment'].str.contains("^\s*\S*[0-9]\S*\s*$"), ['Time-to-edit', 'words']].reset_index(drop=True)
time.columns= ["time (ms)", "words"]

# Join important columns to single dataframe
df = pd.concat([ter, time], axis=1)

# Calculate translation rate (and normalise)
df['perms'] = df['words'] / df['time (ms)']
df['rate'] = (df['perms'] - df['perms'].min()) / (df['perms'].max() - df['perms'].min())

# Remove outliers
df = remove_outliers(df, 'rate', lq=0.05, uq=0.95)

# Correlation
print(df.corr().round(3)['score'])