#!/usr/bin/bash

set -eux

export CPATH=$PWD/secp256k1-install/include:$CPATH
export LIBRARY_PATH=$PWD/secp256k1-install/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$PWD/secp256k1-install/lib:$LD_LIBRARY_PATH

make -j

time ./vanity

./chk

cat 
