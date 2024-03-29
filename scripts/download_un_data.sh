#!/bin/bash

# wget -O $2 'https://daccess-ods.un.org/access.nsf/GetFile?Open&DS='$1'&Lang=E&Type=DOC'

# $1 is the document code
# $2 is the desired output name (downloads a compressed xml file unless .docx is specified)

# The codes are the same as the file names, except replacing / with _).  
# Replace Lang=E with Lang=F or Lang=S to download translations. 

# download documents as docx (from a list of codes)
while IFS= read -r line; do
    wget -O ./data/un_texts/en-es/$line.docx 'https://daccess-ods.un.org/access.nsf/GetFile?Open&DS='$line'&Lang=S&Type=DOC'
done < data/un_texts/un-reliability.ids

# Convert to txt:
# libreoffice --headless --convert-to "txt:Text (encoded):UTF8" $1

for filename in ./data/un_texts/en-es/*.docx; do
    libreoffice --headless --convert-to "txt:Text (encoded):UTF8" $filename
done

# # obtaining texts (codes from filenames in directory)
# for filename in ./data/timed-un/un-readability/*.txt; do 
#     code=$(echo "$(basename $filename .txt)" | tr '_' '/')
#     wget -O ./data/un_texts/en-es/$(basename $filename .txt).docx 'https://daccess-ods.un.org/access.nsf/GetFile?Open&DS='$code'&Lang=S&Type=DOC'
# done