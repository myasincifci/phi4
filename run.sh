#!/bin/bash
#SBATCH --job-name=phi-4
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run --nv -B /tmp/camelyon17_v1.0.sqfs:/data/camelyon17_v1.0:image-src=/ \
    ../../containers/dispatch-new.sif \
    python \
        train.py \
            --config-name bt_disc_ft_freeze1-3