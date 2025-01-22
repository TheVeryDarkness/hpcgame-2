#!/bin/zsh
set -eux

cd handout

cmake -B build
cmake --build build
time ./build/program

cd -

cd handin

cmake -B build
cmake --build build
time ./build/program

cd -
