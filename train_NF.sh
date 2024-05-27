#!bin/sh
#PBS -N yeast_simulation_NF
#PBS -l walltime=2:00:00
#PBS -l nodes=1:ppn=1:gpus=1

python -u train_NF.py --lr 1e-4 --param_l1_loss 1 --var_loss 10 --mean_loss 1 --log-every 5 --check-every 60