set -eux

g++ -O3 -march=native -fopenmp program.cpp -o program
time OMP_NUM_THREADS=2 ./program
