#!/bin/bash
 
filename="backup"
filename2="tst.txt"

echo "Something_${filename}_something"

# check if I can overwrite the lines with a 
# variable
line_number=2
replacement="This line was replaced"
sed -i  "2s/.*/${replacement}/" ${filename2}

