#!binbash

DATA=/rc_scratch/adwi9965/sig-2017-inflection

for l in $(ls ${DATA} | cut -d'-' -f1 | sort | uniq); do
    sbatch --export=ALL,language=${l} curc_exps/train_one_decoder_only.sh
done