#!/bin/sh
#
#SBATCH --account=pimri         # Please stick to this user name
#SBATCH --job-name=Jimmy     # The job name.
#SBATCH --time=20:00:00         # Maximum 120 hours = 5 day
#SBATCH -c 12                   # Number of CPU cores in use. 24CPUs are the maximum number for a single node
#SBATCH --gres=gpu:1            # Request GPU units (4 maximum for K80, 2 maximum for P100)
#SBATCH --mem=64gb               # The memory the job will use per node (128 maximum for one node).


# Setup Environment
module load cuda90/toolkit
module load cuda90/blas
module load cudnn/7.0.5
module load anaconda 

#Command to execute Python program
python eeg_main2.py
