import re
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("data/en-es.pe", sep='\t').drop_duplicates()

# Sentence word count
# df['words'] = 0
# for i in df.index:
#     df.iloc[i,-1] = len(df['Segment'][i].split())

#plt.scatter(df['words'], df['Time-to-edit'])
#a = df.loc[(df['words']>4) & (df['words']<15)]

en = df['Segment'].copy()
mt = df['Suggestion'].copy()
ht = df['Translation'].copy()

assert len(ht) == len(mt), "length of ht and mt are not equal"

def pre_processing(ht, mt):

    print("processing %d sentences..." % len(ht))

    for i in range(0, len(ht)):
        
        # remove <g ids>   
        ht.iloc[i] = re.sub('<g id=\"[0-9]\">', "", ht.iloc[i])
        mt.iloc[i] = re.sub('<g id=\"[0-9]\">', "", mt.iloc[i])

        # remove </g>   
        ht.iloc[i] = re.sub('</g>', "", ht.iloc[i])
        mt.iloc[i] = re.sub('</g>', "", mt.iloc[i])
        
        # remove quotation marks
        ht.iloc[i] = re.sub("[«,»]", "", ht.iloc[i])
        mt.iloc[i] = re.sub("&quot;", "", mt.iloc[i])      

        # remove remaining digits
        ht.iloc[i] = re.sub("\S*[0-9]\S*", "", ht.iloc[i])
        mt.iloc[i] = re.sub("\S*[0-9]\S*", "", mt.iloc[i])   

        # Add sentence number for TER score
        ht.iloc[i] = ht.iloc[i]+ " (A-"+str(i)+")"
        mt.iloc[i] = mt.iloc[i]+ " (A-"+str(i)+")"

    # Remove empty lines:
    ht = ht[~ht.str.contains("^\s*\(A-[0-9]+\)$")]
    mt = mt[~mt.str.contains("^\s*\(A-[0-9]+\)$")]

    return ht, mt

ht, mt = pre_processing(ht, mt)

# Save outputs to new file
with open("ht.txt", 'w') as f:
	for item in ht:
		f.write("%s\n" % item)
print("Human translation saved to ht.txt")

with open("mt.txt", 'w') as f:
	for item in mt:
		f.write("%s\n" % item)
print("Machine translation saved to mt.txt")