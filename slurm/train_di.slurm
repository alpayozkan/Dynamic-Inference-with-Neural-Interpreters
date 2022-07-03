#!/bin/bash

#SBATCH --nodes=1
#sbatch --nodelist=romer1
#SBATCH --job-name=dint
#SBATCH --workdir=/home/alpayozkan/
#SBATCH --gres=gpu:1
#SBATCH --output=/home/alpayozkan/exp/dint-%j.out                  # output file
#SBATCH --error=/home/alpayozkan/exp/dint-%j.err                   # error file
#SBATCH --time=2-00:00:00                                                               # max job time. (currently no need to use)


########### Cuda paths, you may not need that #############
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
#########################################################

dir=/home/alpayozkan/.git/Dynamic-Inference-with-Neural-Interpreters

###### train model
python3 $dir/train_vision.py
which nvcc

###### success message
echo "JOB DONE!!!"
