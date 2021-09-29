#!/bin/bash
#SBATCH -N 1                # request one node
#SBATCH -t 96:00:00	        # request two hours
#SBATCH -p single          # in single partition (queue)
#SBATCH -A hpc_cdss_05
#SBATCH --ntasks-per-node=48
#SBATCH -o logs/slurm-%N.out-%j # optional, name of the stdout, using the job number (%j) and the hostname of the node (%N)
#SBATCH -e logs/slurm-%N.err-%j # optional, name of the stderr, using job and hostname values

# below are job commands
date

# Set some handy environment variables.

export HOME_DIR=/home/$USER/
export WORK_DIR=/work/$USER/deep-tune/
export NCCL_DEBUG=INFO 
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1

# Make sure the WORK_DIR exists:
#mkdir -p $WORK_DIR

# execute code
#time python -u main-rand.py &> ../analysis/output/sgl-tune-rand.txt
#time python -u main-asha.py &> ../analysis/output/sgl-tune-asha.txt
time python -u main.py &> ../analyses/output/tune-alex-asha.txt

# Mark the time it finishes.
date
# exit the job
exit 0
