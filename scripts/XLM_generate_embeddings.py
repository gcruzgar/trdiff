#!/usr/bin/env python3

import argparse
import pandas as pd

""" Modified from XLM repo """
import os
import torch

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

def main():

	# Load pre-trained model
	model_path = './models/mlm_tlm_xnli15_1024.pth'
	reloaded = torch.load(model_path)
	params = AttrDict(reloaded['params'])

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
	#model.cuda() #if using GPU
	model.load_state_dict(reloaded['model'])
	""" """

	with open(args.filename, "r") as f:
		sentence_list=f.readlines()[args.sn[0]:args.sn[1]] 

	# remove new line symbols
	for i in range(0, len(sentence_list)):
		sentence_list[i] = sentence_list[i].replace("\n", "")

	# save as dataframe and add language tokens
	sentence_df = pd.DataFrame(sentence_list)
	sentence_df.columns = ['sentence']
	sentence_df['language'] = 'en'
	
	# match xlm format
	sentences = list(zip(sentence_df.sentence, sentence_df.language))  (sentence, language)

	""" from XLM repo """
	# add </s> sentence delimiters
	sentences = [(('</s> %s </s>' % sent.strip()).split(), lang) for sent, lang in sentences]

	# Create batch
	bs = len(sentences)
	slen = max([len(sent) for sent, _ in sentences])
	
	word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
	for i in range(len(sentences)):
		sent = torch.LongTensor([dico.index(w) for w in sentences[i][0]])
		word_ids[:len(sent), i] = sent
	
	lengths = torch.LongTensor([len(sent) for sent, _ in sentences])
	langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs)
	
	#if using GPU:
	#word_ids=word_ids.cuda()
	#lengths=lengths.cuda()
	#langs=langs.cuda()
		
	tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()
	print(tensor.size())

	# The variable tensor is of shape (sequence_length, batch_size, model_dimension).
	# tensor[0] is a tensor of shape (batch_size, model_dimension) that corresponds to the first hidden state of the last layer of each sentence.
	# This is this vector that we use to finetune on the GLUE and XNLI tasks.
	""" """

	torch.save(tensor[0], args.o)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('filename', type=str, nargs='?',
		help="file containing sentences (including path from current directory)")
	parser.add_argument('-sn', type=int, nargs=2, default = [0,500],
		help="index of sentences to process")
	parser.add_argument('-o', type=str, nargs='?', default="xlm-embeddings.pt",
		help="specify output filename")
	args = parser.parse_args()

	main()

