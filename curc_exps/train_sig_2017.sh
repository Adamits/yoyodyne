DATA=/rc_scratch/adwi9965/sig-2017-inflection
languages=$(ls ${DATA} | cut -d'-' -f1 | sort | uniq)

for l in languages; do
    sbatch curc_exps/train_one_decoder_only.sh --export=ALL,language=${language}
