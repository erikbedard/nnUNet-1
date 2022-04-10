#!/bin/bash

# USAGE: decrypt.sh file.tar.gpg password

# remove ".tar.gpg" extensions
base=$(echo "${1%%.*}") 

# decrypt
echo $2 | gpg --passphrase-fd 0 --pinentry-mode loopback -o ./$base.tar -d $base.tar.gpg

# exract tar
tar -xvf ./$base.tar -C ./
rm ./$base.tar