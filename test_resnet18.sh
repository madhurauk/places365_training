#!/bin/bash
#SBATCH --job-name=rn18_365
#SBATCH --output=models/resnet18/accuracy/logs-%j.out
#SBATCH --error=models/resnet18/accuracy/logs-%j.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition=long

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
#python main.py -a resnet18 --dist-url 'tcp://127.0.0.1:8888' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /coc/scratch/mummettuguli3/data/imagenet
# python test.py
for i in {2..90}
do
python test_placesCNN.py --resume "models/resnet18/model_state_epoch_${i}.pt" --evaluate --workers 4 /coc/scratch/mummettuguli3/data/places365_3/places365_standard
done