#!/bin/bash
for n in 30 60; do
  for t in 1 2 4 8; do
    for rep in 1 2 3; do
      echo -n "n=$n threads=$t run=$rep: "
      MAX_DENSE_GB=1 OMP_NUM_THREADS=$t ./cpu_only Matrices/poisson2d_$n.mtx | grep -E "Iterations|Solve Time" | tr '\n' ' '
      echo
    done
  done
done
