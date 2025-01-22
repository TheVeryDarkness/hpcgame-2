set -eux

cd "$(dirname "$0")/handout"
make generate && ./generate 65536	65536 input.dat
