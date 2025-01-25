set -eux

mpiicpx forest.cpp -O3 -xHost -o forest

export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=core  
mpirun -machinefile hostfile.txt -np 64 ./forest forest.in forest.out
