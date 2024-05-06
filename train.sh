#!bin/sh
#PBS -N yeast_simulation
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=1:gpus=1

python -u main.py --lr 5e-4 --loss Evidential --log-every 40 --check-every 720
# python -u new_MLP_Yeast_dropout.py 