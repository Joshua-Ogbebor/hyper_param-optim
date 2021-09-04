#!/bin/bash
#SBATCH -N 4                	# request two nodes
#SBATCH -n 32 		       	# specify 16 MPI processes (8 per node)
#SBATCH -c 6			# specify 6 threads per process
#SBATCH -t 24:00:00
#SBATCH -p checkpt
#SBATCH -A hpc_cdss_05
#SBATCH -o logs/slurm-%j.out-%N # optional, name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH -e logs/slurm-%j.err-%N # optional, name of the stderr, using job and first node values
# below are job commands
date

# Set some handy environment variables.

export HOME_DIR=/home/$USER/myjob
export WORK_DIR=/work/$USER/myjob
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

export NCCL_SOCKET_IFNAME==^docker0,lo
export PYTHONFAULTHANDLER=1


export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
#export NCCL_MIN_NCHANNELS

# load appropriate modules, in this case Intel compilers, MPICH

# Make sure the WORK_DIR exists:
# mkdir -p $WORK_DIR
# Copy files, jump to WORK_DIR, and execute a program called "my_mpi_demo"
# cp $HOME_DIR/mydemo $WORK_DIR
# cd $WORK_DIR
srun -N4 -n32 -c6 main_asha.py &> "multi_tune_asha_out.txt" # Launch the MPI application with two nodes, 8 MPI processes each node, and 6 threads per MPI process.
# Mark the time it finishes.
date
# exit the job
exit 0
