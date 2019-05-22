# sed to change full stops to fullstop newline and sample text (A1):
# sed 's/\./.\n\(A1\)/g' input.txt > output.txt

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