#!/bin/bash

# USAGE: source encrypt.sh folder password

tar -cvf $1.tar $1
echo $2 | gpg --passphrase-fd 0 --pinentry-mode loopback --symmetric $1.tar
rm $1.tar