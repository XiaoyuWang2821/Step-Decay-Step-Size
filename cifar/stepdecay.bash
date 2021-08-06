#!/usr/bin/env bash
#SBATCH -A SNIC2020-33-39 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=V100:1
#SBATCH -t 48:00:00
#SBATCH -e error_stepdecay.e
#SBATCH -o output_stepdecay.o
#nvidia-smi

module load GCC/8.3.0 CUDA/10.1.243 OpenMPI/3.1.4 PyTorch/1.6.0-Python-3.7.4 tqdm
module load scipy/1.4.1-Python-3.7.4 torchvision/0.7.0-Python-3.7.4-PyTorch-1.6.0
#beta = 1
#CUDA_VISIBLE_DEVICES=0
#print('beta', beta )

python main.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.66667  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data
