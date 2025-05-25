#!/bin/bash

# run on 1 machine * 28 process, feel free to change it!
# srun -N 1 -n 28 $*
if [ "$2" -le 100 ]; then
  srun -N 1 -n 1 --cpu-bind=cores --exclusive $*
elif [ "$2" -le 1000 ]; then
  srun -N 1 -n 2 --cpu-bind=cores --exclusive $*
elif [ "$2" -le 10000 ]; then
  srun -N 1 -n 20 --cpu-bind=cores --exclusive $*
elif [ "$2" -le 50000 ]; then
  srun -N 1 -n 28 --cpu-bind=cores --exclusive $*
else
  srun -N 2 -n 56 --cpu-bind=cores --exclusive $*
fi