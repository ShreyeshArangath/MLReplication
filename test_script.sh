#!/bin/sh
#$ -V
#$ -cwd
#$ -S /bin/bash
#$ -N TEST10K36T
#$ -o $JOB_NAME.o$JOB_ID
#$ -e $JOB_NAME.e$JOB_ID
#$ -q omni
#$ -M shreyesh.arangath@ttu.edu
#$ -m beas
#$ -pe sm 36
#$ -l h_vmem=5.3G
#$ -l h_rt=48:00:00
#$ -P quanah
python3 RFWithSingleDimensionalFeature.py
