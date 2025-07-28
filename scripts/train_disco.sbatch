#!/bin/bash

#SBATCH --job-name=encodec_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --time=96:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,h100_1,v100,rtx8000

# Singularity path
ext3_path=/scratch/$USER/python11/overlay-25GB-500K.ext3
sif_path=/scratch/lg154/python11/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

# start running
singularity exec --nv --overlay /scratch/lg154/python11/overlay-25GB-500K.ext3:ro \
/scratch/lg154/python11/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh
export SSL_CERT_FILE=/scratch/lg154/sseg/fs-ood/cacert.pem
export CUDA_VISIBLE_DEVICES=0

python scripts/train_disco.py --args.load conf/disco_base.yml --save_path runs/test/

"
