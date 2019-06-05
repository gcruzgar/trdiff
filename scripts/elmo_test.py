#!/usr/bin/env python3

from allennlp.commands.elmo import ElmoEmbedder
import scipy

elmo = ElmoEmbedder()
tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
vectors = elmo.embed_sentence(tokens)

assert(len(vectors) == 3) # one for each layer in the ELMo output
assert(len(vectors[0]) == len(tokens)) # the vector elements correspond with the input tokens


vectors2 = elmo.embed_sentence(["I", "ate", "a", "carrot", "for", "breakfast"])

""" UN corpus """
# # Load data
# filename = "data/en-fr-100.en"
# with open(filename, "r") as f:
#     en=f.readlines()

# # Unique tokens in data
# tokens = []
# for sentence in en:
#     words=sentence.split()    
#     tokens.extend(set(words))

# elmo = ElmoEmbedder()
# vectors = elmo.embed_sentence(tokens)

# assert(len(vectors) == 3) # one for each layer in the ELMo output
# assert(len(vectors[0]) == len(tokens)) # the vector elements correspond with the input tokens

# sentence_vectors=[]
# for sentence in en:
#     sentence_vectors.append(elmo.embed_sentence(sentence.split()))
