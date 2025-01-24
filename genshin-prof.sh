set -eux

g++ -std=c++17 -pg -Og -march=native genshin.cpp -o genshin
gprof -b ./program >report.txt
python3 genshin-check.py

prof2dot -f gprof report.txt | dot -Tpng -o output.png

