import re
import os

import pandas as pd

def filter_alignment(i=0, min_alignment=0.75):

    data_path = "data/un_texts/alignments/"
    files = os.listdir(data_path)

    filename = files[i]

    aln = pd.read_csv(data_path+filename, sep='\t', header=None).dropna(axis=0, how='any')
    aln.columns = ['Original', 'Translation', 'Alignment']

    filtered_aln = aln.loc[aln['Alignment'] > min_alignment]


ALIGNMENT = "data/un_texts/alignments/"
ORIGINAL = "data/un_texts/en/"
SPANISH_HT = "data/un_texts/en-es/"
SPANISH_MT = "data/un_texts/en-es-mt.processed"

with open(SPANISH_MT, 'r') as f:
    mt = f.readlines()

# processing on mt document:
#sed 's/<unk>//g' en-es-un-reliability-sent.mt > en-es-mt.processed  # Remove <unk>
#sed -i 's/ p //g' en-es-mt.processed                                # Remove p used for formatting
#sed -i 's/( *,* *,* *)//g' en-es-mt.processed                       # Remove empty brackets
#sed -i 's/^ *\. *$//g' en-es-mt.processed                           # Remove lines only containing full stop
#sed -i '/^[[:space:]]*$/d' en-es-mt.processed                       # Remove empty lines

""" """

filename = "A_HRC_WG.6_24_EST_1.es.mt"
with open(filename, 'r') as f:
    mt = f.readlines()

filename = "A_HRC_WG.6_24_EST_1.es"
with open(filename, 'r', encoding='utf-8-sig') as f:
    ht = f.readlines()

for i in range(0, len(mt)):
    mt[i] = re.sub("<unk>", "", mt[i])
    mt[i] = re.sub("\sp\s", "\n", mt[i])


#sed 's/\sp\s/\n/g' A_HRC_WG.6_24_EST_1.es.mt > test.mt
#sed 's/<unk>//g' test.mt > test2.mt #might need to do this after allignment.
#sed '/^[[:space:]]*$/d' test2.mt > test3.mt

#sed 's/\sp\s/\n<p>\n/g' A_HRC_WG.6_24_EST_1.fr.mt > test.mt
#src/hunalign/hunalign -realign en-fr-cognates.txt A_HRC_WG.6_24_EST_1.en A_HRC_WG.6_24_EST_1.fr.mt > fr.aln

#sed 's/p\s<unk>\sp\(\s<unk>\sp\)*/\n<p>\n/g'

#sed '/original\s<unk>\sanglais/q' #output text up to match "original <unk> anglais"

filename="en-fr.un-reliability-sent.mt"
with open(filename, 'r') as f:
    mt_text = f.readlines()

delim_list = []
for i in range(0, len(mt_text)):
    if re.search("original <unk> anglais", mt_text[i]):
        delim_list.append(i)
