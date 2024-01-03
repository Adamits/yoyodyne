#SBATCH --mail-user=adwi9965@colorado.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=20gb
#SBATCH --time=24:00:00
#SBATCH --qos=preemptable
#SBATCH --constraint=V100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_one_inflection.%j.log

source /curc/sw/anaconda3/latest
conda activate yoyodyne

# TASK
# train_filename
# dev_filename

TASK=sig-2023
train_filename=deu.trn
dev_filename=deu.dev
# Fixes path issue.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/adwi9965/software/anaconda/envs/yoyodyne/lib/
readonly ROOT="/projects/adwi9965/yoyodyne"
readonly DATA="/rc_scratch/adwi9965/${TASK}"

readonly TRAIN="${DATA}/${train_filename}"
readonly DEV="${DATA}/${dev_filename}"

readonly RESULTS_PATH=/rc_scratch/adwi9965/decoder-only-results/${train_filename}-sweep

readonly OPTIMIZER=adam
readonly LEARNING_RATE=0.001
readonly BETA2=0.98
readonly BATCH_SIZE=400
readonly DROPOUT=0.3
readonly ENCODER_LAYERS=2
# readonly DECODER_LAYERS=8
# readonly ATTENTION_HEADS=4
# readonly EMBEDDING_SIZE=256
# readonly HIDDEN_SIZE=1024
readonly ARCH=decoder_only_transformer

# Print GPU topology if gpus are present
nvidia-smi || true
nvidia-smi topo -m || true

for embedding_size in 64 128 256 512; do
    for hidden_size in 128 256 512 1024; do
        for num_layers in 2 4 8; do
            for num_heads in 1 2 4 8; do
            for warmup in 2000 4000; do
                yoyodyne-train \
                    --arch "${ARCH}" \
                    --accelerator gpu \
                    --batch_size "${BATCH_SIZE}" \
                    --target_col ${TARGET_COL} \
                    --features_col ${FEATURES_COL} \
                    --experiment ${TASK}-decoder-only-deu-sweep \
                    --train "${TRAIN}" \
                    --val "${DEV}" \
                    --max_epochs 800 \
                    --learning_rate "${LEARNING_RATE}" \
                    --embedding_size "${embedding_size}" \
                    --hidden_size "${hidden_size}" \
                    --dropout "${DROPOUT}" \
                    --encoder_layers 1 \
                    --decoder_layers "${num_layers}" \
                    --source_attention_heads "${num_heads}" \
                    --label_smoothing 0.1 \
                    --optimizer "${OPTIMIZER}" \
                    --beta2 "${BETA2}" \
                    --scheduler warmupinvsqrt \
                    --warmup_steps "${warmup}" \
                    --save_top_k 1 \
                    --check_val_every_n_epoch 16 \
                    --model_dir "${RESULTS_PATH}" \
                    --seed 42 \
                    --log_wandb \
                    --hierarchical_features ;
                done;
            done;
        done;
    done;
done;