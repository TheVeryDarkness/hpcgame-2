set -eux

g++ -std=c++17 -pg -Og -march=native genshin.cpp -o genshin
gprof -b ./genshin gmon.out > gprof.txt
python3 genshin-check.py

prof2dot -f gprof gprof.txt | dot -Tpng -o output.png

