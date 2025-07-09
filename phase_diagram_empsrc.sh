#!/usr/bin/bash

D=$1
L=$2
I=$3
NOISE=$4
SEED=$5
GSRC=$6

DIR_NAME=l=${L}-d=${D}/noise=${NOISE}/


if [ ! -d ../${DIR_NAME} ]; then
    mkdir -p ../${DIR_NAME}
fi

GAMMA=$(seq 0.1 0.1 2)
N=$(echo $GAMMA | awk -v d="$D" '{for (i=1; i<=NF; i++) printf "%d ", int($i * d + 0.5)}')
NSRC=$(awk "BEGIN {print int(($GSRC * $D) + 0.5)}")

for THETA in $(seq 0.05 0.05 1)
do 
    python3 ../transfer.py --layers $L \
                           --d $D \
                           --instances $I \
                           --init_scale 1e-5 \
                           --save_path ../$DIR_NAME \
                           --theta $THETA \
                           --epochs 8000 \
                           --tol 1e-6 \
                           --save_freq 100 \
                           --noise $NOISE \
                           --noise_src $NOISE \
                           --n_tar $N \
                           --n_src $NSRC \
                           --fine_tune \
                           --linear_transfer \
                           --lt_reg 0 0.1 0.2 0.3 0.4 0.5 \
                           --device gpu \
                           --seed $SEED \
                           --exp_name n_src-theta-seed
                        

done



