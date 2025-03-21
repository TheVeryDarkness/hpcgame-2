#!/usr/bin/bash

set -eux

cd secp256k1
# chmod +x autogen.sh
bash ./autogen.sh
export CFLAGS="-O3 -march=native -fPIC"
export CXXFLAGS="-O3 -march=native -fPIC"
./configure --prefix=$PWD/../secp256k1-install
make -j$(nproc)
make check -j
make install
cd ..
