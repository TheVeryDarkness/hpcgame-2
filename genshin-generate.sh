set -eux

python3 genshin-generate.py

clang++ -std=c++20 -O3 -march=native genshin_1.cpp -o genshin_1
time ./genshin_1
