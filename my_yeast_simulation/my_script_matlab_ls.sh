#!/bin/bash
#PBS -l walltime=24:00:00
#PBS -l nodes=1:ppn=28
#PBS -A PAS1282

cd $HOME/Random_runs/
module load matlab
matlab -nojvm < least_sq5.m
