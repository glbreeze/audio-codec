#!/bin/bash

#SBATCH --job-name=encodec_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --account=pr_119_tandon_priority

FILM=$1
ALIGN=$2
HBT=$3
RVQ=$4

# Singularity path
ext3_path=/scratch/lb4434/ML2/overlay-50G-10M.ext3
sif_path=/scratch/lb4434/ML2/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

# start running
singularity exec --nv --overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
conda activate vq
cd /scratch/lb4434/audio-codec/

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/scratch/lb4434/audio-codec:$PYTHONPATH

python scripts/train_disco.py --args.load conf/disco_base.yml --save_path runs/libri_film${FILM//,/}_align${ALIGN}_hbt${HBT}_size${RVQ}/ \
    --lambdas.align/loss ${ALIGN} --DiscoDAC.film_layers \"[${FILM}]\" --exp_name film${FILM//,/}_align${ALIGN}_size${RVQ} --hubert_layer ${HBT} --no-capture-output \
    --DiscoDAC.n_codebooks ${RVQ}
"
