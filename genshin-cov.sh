set -eux

clear

rm program.gcda || true
g++ -std=c++17 -fprofile-arcs -ftest-coverage --coverage -g -Og -march=native program.cpp -o program
OMP_NUM_THREADS=2 ./program
diff out.data out.data.std
gcov -f ./program >./report.gcov.txt
