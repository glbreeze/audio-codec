#!/bin/bash

#SBATCH --job-name=probe_asr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,h100_1,v100,rtx8000


# start running
singularity exec --nv --overlay /scratch/lg154/python_3D/overlay-25GB-500K.ext3:ro \
    /scratch/lg154/python_3D/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        export SSL_CERT_FILE=/scratch/lg154/sseg/fs-ood/cacert.pem
        export PYTHONPATH=$PWD:$PYTHONPATH

        echo 'Running Probe ASR on DAC codes with all codes...'

        python scripts/wav_eval_code.py \
            --data_root runs_08/cb3_baseline/asr_data/ \
            --input_type discrete \
            --epochs 100 \
            --n_codebooks -1 \
            --exp_name dac_all
        
        echo 'Running Probe ASR on DiscoDAC codes with all codes...'

        python scripts/wav_eval_code.py \
            --data_root runs_08/cb2_film03_align1/asr_data/ \
            --input_type discrete \
            --epochs 100 \
            --n_codebooks -1 \
            --exp_name disco_all
    "
