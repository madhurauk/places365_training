#!/bin/bash
#SBATCH --job-name=gc_in_rn50_resume
#SBATCH --output=GRADCAM_MAPS/resnet50/logs-%j.out
#SBATCH --error=GRADCAM_MAPS/resnet50/logs-%j.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition=long

source /nethome/mummettuguli3/anaconda2/bin/activate
conda activate my_basic_env_3
for i in {1..90}
do
python test_2_gc_resnet50.py -a resnet50 --resume "models/resnet50/model_state_epoch_${i}.pt" --evaluate --output_dir "GRADCAM_MAPS/resnet50/${SLURM_JOBID}" --workers 4 /coc/scratch/mummettuguli3/data/places365_3/places365_standard --class_list airfield art_gallery
done

python create_gif.py --output_dir "GRADCAM_MAPS/resnet50/${SLURM_JOBID}/" --dataset "places365" --class_list airfield art_gallery