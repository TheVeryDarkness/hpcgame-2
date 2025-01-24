set -eux

clear

g++ -std=c++17 -g -Og -march=native program.cpp -o program
export OMP_NUM_THREADS=2 
valgrind --tool=callgrind ./program
diff out.data out.data.std
