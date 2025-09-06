#!/bin/bash

#SBATCH --job-name=encodec_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,h100_1,v100,rtx8000

CB=$1
FILM=$2
ALIGN=$3

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

python scripts/train_disco.py --args.load conf/disco_base.yml \
    --hubert_layer 9 \
    --lambdas.align/loss 1 \
    --SemanticEmbedLoss.loss_type l2 \
    --DiscoDAC.n_codebooks ${CB} \
    --DiscoDAC.film_layer_idx ${FILM} \
    --DiscoDAC.large_kernel f \
    --DiscoDAC.latent_dim 1024 \
    --batch_size 48 \
    --save_path runs_08/cb${CB}_film${FILM}_align${ALIGN}_ld48/ \
    --exp_name cb${CB}_film${FILM}_align${ALIGN}_ld48
"

# NS=$4
# NF=$5
# --DiscoDAC.new_sem ${NS} \
# --DiscoDAC.new_film ${NF} \
# --save_path runs_08/cb${CB}_film${FILM}_LK${LK}_NS${NS}_NF${NF}/ \
# --exp_name cb${CB}_film${FILM}_LK${LK}_NS${NS}_NF${NF}