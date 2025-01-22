set -eux

cd "$(dirname "$0")/handout"
make lcs && OMP_NUM_THREADS=16 ./lcs
