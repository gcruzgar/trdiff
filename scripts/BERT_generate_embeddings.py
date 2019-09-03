#!/usr/bin/env python3

""" Generating sentence embeddings using pre-trained BERT models. 
This script uses https://github.com/hanxiao/bert-as-service for a 
simpler sentence vectorisation
"""


# Initialise server from terminal (can specify model)
#bert-serving-start -model_dir data/bert_models/uncased_L-12_H-768_A-12 -num_worker=1 -max_seq_len=128

import numpy as np
import pandas as pd 

from bert_serving.client import BertClient

lan = 'es'

# Load sentences
wpd = pd.read_csv("data/un-timed-sentences/en-"+lan+".processed", sep='\t').drop_duplicates()
sentences = list(wpd['Segment'])

bc = BertClient()
df = pd.DataFrame(bc.encode(sentences))

df.to_csv("bert-embeddings-timed-sentences-"+lan+".csv", header=None, index=False)