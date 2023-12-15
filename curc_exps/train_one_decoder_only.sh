#!/bin/bash

#SBATCH --mail-user=adwi9965@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=20gb
#SBATCH --time=24:00:00
#SBATCH --qos=preemptable
#SBATCH --constraint=V100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/decoder_only_inflection.%j.log

source /curc/sw/anaconda3/latest
conda activate yoyodyne

# SET `language` as env variable when calling with sbatch.

# Fixes path issue.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/adwi9965/software/anaconda/envs/yoyodyne/lib/
readonly TASK="sig-2017-inflection"
readonly ROOT="/projects/adwi9965/yoyodyne"
readonly DATA="/rc_scratch/adwi9965/${TASK}"

readonly TRAIN="${DATA}/${language}-train-high"
readonly DEV="${DATA}/${language}-dev"

readonly RESULTS_PATH=/rc_scratch/adwi9965/decoder-only-results

readonly OPTIMIZER=adam
readonly LEARNING_RATE=0.001
readonly BETA2=0.98
readonly BATCH_SIZE=400
readonly DROPOUT=0.3
readonly ENCODER_LAYERS=2
readonly DECODER_LAYERS=8
readonly ATTENTION_HEADS=4
readonly EMBEDDING_SIZE=256
readonly HIDDEN_SIZE=1024
readonly ARCH=decoder_only_transformer

# Print GPU topology if gpus are present
nvidia-smi || true
nvidia-smi topo -m || true

yoyodyne-train \
    --arch "${ARCH}" \
    --accelerator gpu \
    --batch_size "${BATCH_SIZE}" \
    --target_col 2 \
    --features_col 3 \
    --experiment ${language}-decoder-only \
    --train "${TRAIN}" \
    --val "${DEV}" \
    --max_epochs 800 \
    --learning_rate "${LEARNING_RATE}" \
    --embedding_size "${EMBEDDING_SIZE}" \
    --hidden_size "${HIDDEN_SIZE}" \
    --dropout "${DROPOUT}" \
    --encoder_layers "${ENCODER_LAYERS}" \
    --decoder_layers "${DECODER_LAYERS}" \
    --source_attention_heads "${ATTENTION_HEADS}" \
    --label_smoothing 0.1 \
    --optimizer "${OPTIMIZER}" \
    --beta2 "${BETA2}" \
    --scheduler warmupinvsqrt \
    --warmup_steps 4000 \
    --save_top_k 1 \
    --check_val_every_n_epoch 16 \
    --model_dir "${RESULTS_PATH}" \
    --seed 42 \
    --log_wandb;