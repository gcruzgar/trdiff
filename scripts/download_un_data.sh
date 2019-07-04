#!/bin/bash

# wget -O $2 'https://daccess-ods.un.org/access.nsf/GetFile?Open&DS='$1'&Lang=E&Type=DOC'

# $1 is the document code
# $2 is the desired output name (dowloads a zip file)

# The codes are the same as the file names, except replacing / with _).  
# Replace Lang=E with Lang=F or Lang=S to download translations. 

for filename in ./data/timed-un/un-readability/*.txt; do 
    code=$(echo "$(basename $filename .txt)" | tr '_' '/')
    wget -O ./data/un_texts/en-es/$(basename $filename .txt).docx 'https://daccess-ods.un.org/access.nsf/GetFile?Open&DS='$code'&Lang=S&Type=DOC'
done

# Convert to txt:
# libreoffice --headless --convert-to "txt:Text (encoded):UTF8" $1

for filename in ./data/un_texts/en-es/*.docx; do
    libreoffice --headless --convert-to "txt:Text (encoded):UTF8" $filename
done
