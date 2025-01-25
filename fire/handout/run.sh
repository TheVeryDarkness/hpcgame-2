. /etc/profile >/dev/null #加载环境

set -eux

krun init >/dev/null
krun host -n -H > hostfile

mpiicpx forest.cpp -O3 -xHost -o forest

mpirun --hostfile hostfile -np 4 -ppn 16 /data/forest /problem/input.dat /data/output.dat

# export I_MPI_PIN=1
# export I_MPI_PIN_DOMAIN=core  
# mpirun -machinefile hostfile.txt -np 64 ./forest forest.in forest.out
