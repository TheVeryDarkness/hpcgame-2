. /etc/profile >/dev/null #加载环境

set -eux

krun init >/dev/null
krun host -n -H > hostfile

mpiicpx main.cpp -Wall -O3 -xHost -o forest
# mpiicpx main.cpp -Og -g -xHost -o forest

scp /data/forest root@hpcgame-test-fire-workers-0-1:/data/forest
scp /data/forest root@hpcgame-test-fire-workers-0-2:/data/forest
scp /data/forest root@hpcgame-test-fire-workers-0-3:/data/forest

mpirun --hostfile hostfile -np 4 /data/forest /data/input.txt /data/output.dat

# export I_MPI_PIN=1
# export I_MPI_PIN_DOMAIN=core  
# mpirun -machinefile hostfile.txt -np 64 ./forest forest.in forest.out
