set -eux

# g++ -O3 -march=native -fopenmp program.cpp -o program
# time OMP_NUM_THREADS=2 ./program

clear
clang++ -std=c++20 -g -Og -march=native genshin.cpp -o genshin
time ./genshin

python3 genshin-check.py
