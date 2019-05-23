#!/usr/bin/env python3
import pandas as pd  

# read ter output
filename = "data/en-fr-100.pra"
with open(filename, "r") as f:
    ht=f.readlines() 

df = pd.DataFrame(data=ht, columns=['line']) # save as dataframe for ease
df = df.loc[df['line'].str.contains("Score")] # only need lines containing scores
# only need actual score:
df['line'] = df['line'].str.replace("Score: ", "", regex=False)
df['line'] = df['line'].str.replace("\((.*?)\)\\n", "")

df.rename(index=str,columns={'line': 'score'}, inplace=True)
df.reset_index(drop=True, inplace=True)
df['score'] = df['score'].astype('float')

# Save outputs to new file
with open("mt_score.txt", 'w') as f:
    for item in df['score']:
        f.write("%s\n" % item)
