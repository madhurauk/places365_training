#!/bin/bash
#SBATCH --job-name=resnet50
#SBATCH --output=models/resnet50/logs-%j.out
#SBATCH --error=models/resnet50/logs-%j.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --partition=long

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
python train_placesCNN_resnet50.py -a resnet50 --workers 40 /coc/scratch/mummettuguli3/data/places365_3/places365_standard