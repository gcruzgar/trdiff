#!/usr/bin/env python3

# Open human translation
filename1 = "data/en-fr-100.fr"
with open(filename1, "r") as f:
    ht=f.readlines() 

# Open machine translation
filename2 = "data/en-fr-100.mt"
with open(filename2, "r") as f:
    mt=f.readlines() 

# Add full stop to each end of line and sentence serial number
ht_id=[]
mt_id=[]
for i in range(0, len(ht)):
    ht_id.append(ht[i].replace("\n", ". (A-"+str(i)+")"))
    mt_id.append(mt[i].replace("\n", ". (A-"+str(i)+")"))

# Save outputs to new file
with open("ht.txt", 'w') as f:
    for item in ht_id:
        f.write("%s\n" % item)

with open("mt.txt", 'w') as f:
    for item in mt_id:
        f.write("%s\n" % item)

## regex preprocessing:
# (1) change new line to space:
# tr '\n' ' ' < input.txt > output.txt
# (2) add new line after each sentence:
# sed 's/[.!?;] */&\n/g' input.txt > output.txt
# (3) get rid of new lines generated due to '...': 
# sed 's/\.\n\.\n\./\.\.\./g'  ## DOESNT WORK!
# sed '/^\./d' ## this removes fullstops on empty lines
# (4) remove lines starting with number:
# sed '/^[0-9]\+/d' 
# (5) correct instances of new lines caused by acronyms:
# sed 's/^[A-Z]\.\n/&\s/g' ## DOESNT WORK AS INTENDED
# (6) add sentence serial number:
# see python code
# (7) compute translation edit rate: 
# java -jar tercom.7.25.jar -N -n outputfilename -o pra -r ht.txt -h mt.txt

# Faster steps 2-5 (obtained from stackexchange - use at own risk)
# sed -e :1 -e 's/\([.?!]\)[[:blank:]]\{1,\}\([^[:blank:]]\)/\1\ \2/;t1'