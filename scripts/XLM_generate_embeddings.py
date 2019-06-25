#!/usr/bin/env python3

""" from XLM repo """
import os
import torch

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

# load pre-trained model
model_path = './models/mlm_tlm_xnli15_1024.pth'
reloaded = torch.load(model_path)
params = AttrDict(reloaded['params'])
print("Supported languages: %s" % ", ".join(params.lang2id.keys()))

# build dictionary / update parameters
dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
params.n_words = len(dico)
params.bos_index = dico.index(BOS_WORD)
params.eos_index = dico.index(EOS_WORD)
params.pad_index = dico.index(PAD_WORD)
params.unk_index = dico.index(UNK_WORD)
params.mask_index = dico.index(MASK_WORD)

# build model / reload weights
model = TransformerModel(params, dico, True, True)
model.load_state_dict(reloaded['model'])

# list of (sentences, lang)
# sentences = [
#     ('the following secon@@ dary charac@@ ters also appear in the nov@@ el .', 'en'),
#     ('les zones rurales offr@@ ent de petites routes , a deux voies .', 'fr'),
#     ('luego del cri@@ quet , esta el futbol , el sur@@ f , entre otros .', 'es'),
#     ('am 18. august 1997 wurde der astero@@ id ( 76@@ 55 ) adam@@ ries nach ihm benannt .', 'de'),
#     ('اصدرت عدة افلام وث@@ اي@@ قية عن حياة السيدة في@@ روز من بينها :', 'ar'),
#     ('此外 ， 松@@ 嫩 平原 上 还有 许多 小 湖泊 ， 当地 俗@@ 称 为 “ 泡@@ 子 ” 。', 'zh'),
# ]
""" """

import pandas as pd

# load sentences (first 100 due to memory limit)
filename = "en-fr-100/en-fr-100.en"
with open(filename, "r") as f:
    sentence_list=f.readlines()[0:100] 

# remove new line symbols
for i in range(0, len(sentence_list)):
    sentence_list[i] = sentence_list[i].replace("\n", "")

# save as dataframe and add language tokens
sentence_df = pd.DataFrame(sentence_list)
sentence_df.columns = ['sentence']
sentence_df['language'] = 'en'

# match xlm format (sentence, language)
sentences = list(zip(sentence_df.sentence, sentence_df.language)) 

""" from XLM repo """
# add </s> sentence delimiters
sentences = [(('</s> %s </s>' % sent.strip()).split(), lang) for sent, lang in sentences]

# create batch
bs = len(sentences)
slen = max([len(sent) for sent, _ in sentences])

word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
for i in range(len(sentences)):
    sent = torch.LongTensor([dico.index(w) for w in sentences[i][0]])
    word_ids[:len(sent), i] = sent

lengths = torch.LongTensor([len(sent) for sent, _ in sentences])
langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs)


tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
print(tensor.size())

# The variable tensor is of shape (sequence_length, batch_size, model_dimension).
# tensor[0] is a tensor of shape (batch_size, model_dimension) that corresponds to the first hidden state of the last layer of each sentence.
# This is this vector that we use to finetune on the GLUE and XNLI tasks.
""" """
