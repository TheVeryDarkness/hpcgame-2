. /etc/profile >/dev/null #加载环境

set -eux

# python3 ./gencase.py 16384 1000 input.txt

krun init >/dev/null
krun host -n -H > hostfile

mpiicpx main.cpp -Wall -O3 -xHost -o forest
# mpiicpx main.cpp -Og -g -xHost -o forest

# scp /data/forest /data/1.in /data/2.in root@hpcgame-test-fire-workers-0-1:/data/
# scp /data/forest /data/1.in /data/2.in root@hpcgame-test-fire-workers-0-2:/data/
# scp /data/forest /data/1.in /data/2.in root@hpcgame-test-fire-workers-0-3:/data/

mpirun --hostfile hostfile -np 4 /data/forest /data/1-0.in /data/1-0.out
diff ./1-0.out ./1-0.out.std
mpirun --hostfile hostfile -np 4 /data/forest /data/1-1.in /data/1-1.out
diff ./1-1.out ./1-1.out.std
mpirun --hostfile hostfile -np 4 /data/forest /data/1-2.in /data/1-2.out
diff ./1-2.out ./1-2.out.std
mpirun --hostfile hostfile -np 4 /data/forest /data/1-3.in /data/1-3.out
diff ./1-3.out ./1-3.out.std
mpirun --hostfile hostfile -np 4 /data/forest /data/1-4.in /data/1-4.out
diff ./1-4.out ./1-4.out.std

mpirun --hostfile hostfile -np 4 /data/forest /data/1.in /data/1.out
diff ./1.out ./1.out.std
mpirun --hostfile hostfile -np 4 /data/forest /data/2.in /data/2.out
diff ./2.out ./2.out.std
# mpirun --hostfile hostfile -np 4 /data/forest /data/input.txt /data/output.dat

# export I_MPI_PIN=1
# export I_MPI_PIN_DOMAIN=core  
# mpirun -machinefile hostfile.txt -np 64 ./forest forest.in forest.out
