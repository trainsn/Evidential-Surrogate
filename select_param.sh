#!bin/sh
#PBS -N select_param
#PBS -l walltime=0:10:00
#PBS -l nodes=1:ppn=1:gpus=1

python -u select_param.py --seed 1 --dsp 28 --n-candidates 480000 --resume models/model_Evidential_600.pth.tar --lam 25