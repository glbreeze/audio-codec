#!/bin/bash

#SBATCH --job-name=encodec_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,h100_1,v100,rtx8000

FILM=$1
ALIGN=$2
HBT=$3

# Singularity path
ext3_path=/scratch/$USER/python11/overlay-25GB-500K.ext3
sif_path=/scratch/lg154/python11/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

# start running
singularity exec --nv --overlay /scratch/lg154/python11/overlay-25GB-500K.ext3:ro \
/scratch/lg154/python11/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
/bin/bash -c "
source /ext3/env.sh
export SSL_CERT_FILE=/scratch/lg154/sseg/cacert.pem
export CUDA_VISIBLE_DEVICES=0
export NUMBA_DISABLE_CACHE=1

python scripts/train_disco.py --args.load conf/disco_base.yml --lambdas.align/loss ${ALIGN} \
    --DiscoDAC.film_layer_idx ${FILM} --hubert_layer ${HBT} \
    --save_path runs/libri6_film${FILM}_align${ALIGN}_hbt${HBT}/ \
    --exp_name cb6_film${FILM}_align${ALIGN}_hbt${HBT} 

"
