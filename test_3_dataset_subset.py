import os
import sys
sys.path.append('/srv/share3/mummettuguli3/code/')
from utils.grad_cam_caller import GCUtil

valdir = os.path.join("/coc/scratch/mummettuguli3/data/places365_3/places365_standard" , 'val')

gc_util = GCUtil("places365")
gc_util.create_tiny_dataset(valdir, 10)
#gc_util.create_output_folder(output_dir='GRADCAM_MAPS/resnet18/', dataset="imagenet", class_list=["ostrich"], valdir=valdir, sample_count=2)