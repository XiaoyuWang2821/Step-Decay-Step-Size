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

#python main_sgd_stepdecay_2.py --optim-method SGD_Const_Decay --eta0 0.05 --alpha 0.125  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 128 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data
#
#python main_sgd_stepdecay_3.py --optim-method SGD_Const_Decay --eta0 0.01 --alpha 0.125  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 128 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data
#
#

####### best result is achieved at initial stepsize 1 and alpha is 0.33333
python main_mom_stepdecay.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.66667  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python main_mom_stepdecay_2.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.5  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python main_mom_stepdecay_3.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.33333  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data


python main_mom_stepdecay_4.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.25  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python main_mom_stepdecay_5.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.2  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data


python main_mom_stepdecay_6.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.16667  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python main_mom_stepdecay_7.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.14256  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python main_mom_stepdecay_8.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.125 --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python main_mom_stepdecay_9.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.11111  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data


python main_mom_stepdecay_10.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.1  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data


python main_mom_stepdecay_11.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.090909  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

python main_mom_stepdecay_12.py --optim-method SGD_Step_Decay --eta0 1.0 --alpha 0.08333  --momentum 0.0 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data







#######  decay rate

####### 0.05 
#python main_mom_stepdecay.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.66667  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

#python main_mom_stepdecay_2.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.5  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

#python main_mom_stepdecay_3.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.33333  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data


#python main_mom_stepdecay_4.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.25  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

#python main_mom_stepdecay_5.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.2  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data


#python main_mom_stepdecay_6.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.16667 --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

#python main_mom_stepdecay_7.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.14286  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

#python main_mom_stepdecay_8.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.125  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

#python main_mom_stepdecay_9.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.11111  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data


#python main_mom_stepdecay_10.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.1  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data


#python main_mom_stepdecay_11.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.090909  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

#python main_mom_stepdecay_12.py --optim-method SGD_Step_Decay --eta0 0.05 --alpha 0.08333  --nesterov --momentum 0.9 --weight-decay 0.0005 --train-epochs 164 --batchsize 128 --eval-interval 1 --use-cuda --log-folder ./logs/CIFAR100 --dataset CIFAR100 --dataroot ./data

