#!/usr/bin/env python3

import re
import argparse

def main():

	# Open human translation
	filename1 = args.ht
	with open(filename1, "r") as f:
		ht=f.readlines() 

	# Open machine translation
	filename2 = args.mt
	with open(filename2, "r") as f:
		mt=f.readlines() 

	for i in range(0, len(ht)):
		# remove digits
		ht[i] = re.sub("\S*[0-9]\S*", "", ht[i])
		mt[i] = re.sub("\S*[0-9]\S*", "", mt[i])
		
		# remove unknowns    
		mt[i] = re.sub("(\(\s)?\<unk\>(\s\))?", "", mt[i])
		
		# Add sentence number for TER score (not on original doc)
		ht[i] = ht[i].replace("\n", " (A-"+str(i)+")")
		mt[i] = mt[i].replace("\n", " (A-"+str(i)+")")

	# Save outputs to new file
	with open("ht.txt", 'w') as f:
		for item in ht:
			f.write("%s\n" % item)
	print("Human translation saved to ht.txt")

	with open("mt.txt", 'w') as f:
		for item in mt:
			f.write("%s\n" % item)
	print("Human translation saved to mt.txt")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-ht', type=str, nargs='?',
		help="human translation file to process (including path from current directory)")
	parser.add_argument('-mt', type=str, nargs='?',
		help="machine translation file to process (including path from current directory)")
	args = parser.parse_args()

	main()
