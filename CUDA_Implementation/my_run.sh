#!/bin/bash
#SBATCH --job-name=gpu
#SBATCH --output=gpu.out
#SBATCH --error=gpu.err
#SBATCH --nodes=4
#SBATCH --partition=xgpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:tesla:1

# Set stack size (to avoid warning message)
ulimit -s 10240

module load cuda/10.0
if [ "$#" -eq 2 ]; then
    ./DWT $1 $2
elif [ "$#" -eq 4 ]; then
    ./DWT $1 $2 $3 $4
else
   echo "Error, need one of the following initializations"
   echo "./my_run.sh <filepath> <0 (no compression)>"
   echo "./my_run.sh <filepath> <1 (compression)> <quantization_step> <threshold>"
fi
