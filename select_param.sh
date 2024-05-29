#!bin/sh
#PBS -N yeast_simulation_seed0
#PBS -l walltime=0:10:00
#PBS -l nodes=1:ppn=1:gpus=1

python -u select_param.py --seed 1 --dsp 28 --n-samples 50000