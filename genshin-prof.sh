set -eux

g++ -std=c++17 -pg -g -Og -march=native program.cpp -o program
OMP_NUM_THREADS=2 ./program
diff out.data out.data.std
gprof -A -B -C -p -l -I . ./program >./report.txt

# python3 genshin-check.py

prof2dot -f gprof report.txt | dot -Tpng -o output.png

