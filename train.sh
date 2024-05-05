#!bin/sh
#PBS -N yeast_simulation
#PBS -l walltime=2:00:00
#PBS -l nodes=1:ppn=1:gpus=1

python -u main.py --lr 2e-5 --loss Evidential --log-every 40 --check-every 350
# python -u new_MLP_Yeast_dropout.py 