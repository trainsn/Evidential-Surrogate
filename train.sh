#!bin/sh
#PBS -N yeast_simulation_Evidential_active_lambda25
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=1:gpus=1

python -u train.py --seed 0 --dsp 28 --lr 5e-4 --loss Evidential --log-every 10 --check-every 150 --active --lam 25
# python -u train.py --seed 0 --dsp 28 --dropout --lr 5e-4 --loss MSE --log-every 20 --check-every 200
# python -u train.py --seed 0 --dsp 28 --lr 5e-4 --loss MSE --log-every 20 --check-every 200
# python -u train.py --seed 0 --dsp 28 --lr 5e-4 --loss Gaussian --log-every 20 --check-every 200
# python -u new_MLP_Yeast_dropout.py 