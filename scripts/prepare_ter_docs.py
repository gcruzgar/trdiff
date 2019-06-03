#!/bin/env python3
import re

# Open human translation
filename1 = "UNv1.0.6way.es"
with open(filename1, "r") as f:
	es=f.readlines() 

# Open machine translation
filename2 = "UNv1.0.6way.en-es.mt"
with open(filename2, "r") as f:
	mt=f.readlines() 

for i in range(0, len(es)):
	# remove digits
	es[i] = re.sub("\S*[0-9]\S*", "", es[i])
	mt[i] = re.sub("\S*[0-9]\S*", "", mt[i])
	
	# remove unknowns    
	mt[i] = re.sub("(\(\s)?\<unk\>(\s\))?", "", mt[i])
	
	# Add sentence number for TER score (not on original doc)
	es[i] = es[i].replace("\n", " (A-"+str(i)+")")
	mt[i] = mt[i].replace("\n", " (A-"+str(i)+")")

# Save outputs to new file
with open("es.txt", 'w') as f:
	for item in es:
		f.write("%s\n" % item)

with open("mt.txt", 'w') as f:
	for item in mt:
		f.write("%s\n" % item)
