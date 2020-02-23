#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=4571

module unload Python/3.6.4
module load parallel intel/2017.4.196-GCC-6.4.0-2.28  impi/2017.3.196 Python/2.7.14
pip install --user numpy
pip install --user scipy
pip install --user autograd


MY_PARALLEL_OPTS="-N 1 --delay .2 -j $SLURM_NTASKS --joblog parallel-${SLURM_JOBID}.log"
MY_SRUN_OPTS="-N 1 -n 1 --exclusive"
MY_EXEC="python test_alpha.py yacht 1.5 0.001 500 AR 0.0 1.0 {1} 0_1"

parallel $MY_PARALLEL_OPTS srun $MY_SRUN_OPTS $MY_EXEC ::: {0..55}

