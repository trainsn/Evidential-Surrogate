#!bin/sh
#PBS -N yeast_simulation_seed0
#PBS -l walltime=0:10:00
#PBS -l nodes=1:ppn=1:gpus=1

# python -u main.py --seed 0 --dsp 28 --lr 5e-4 --loss Evidential --log-every 20 --check-every 200
# python -u main.py --seed 0 --dsp 28 --dropout --lr 5e-4 --loss MSE --log-every 20 --check-every 200
python -u main.py --seed 0 --dsp 28 --lr 5e-4 --loss MSE --log-every 20 --check-every 200
# python -u new_MLP_Yeast_dropout.py 