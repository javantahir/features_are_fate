#!/usr/bin/bash

D=$1
L=$2
I=$3
NOISE=$4
SEED=$5

DIR_NAME=l=${L}-d=${D}/noise=${NOISE}/scratch/


if [ ! -d ../${DIR_NAME} ]; then
    mkdir -p ../${DIR_NAME}
fi

GAMMA=$(seq 0.1 0.1 2)
N=$(echo $GAMMA | awk -v d="$D" '{for (i=1; i<=NF; i++) printf "%d ", int($i * d + 0.5)}')

python3 ../train.py --layers ${L} \
                    --d ${D} \
                    --instances ${I} \
                    --init_scale 1e-5 \
                    --save_path ../$DIR_NAME \
                    --epochs 10000 \
                    --lr 0.1 \
                    --tol 1e-6 \
                    --save_freq 100 \
                    --noise ${NOISE} \
                    --n ${N} \
                    --reg 0 1e-4 1e-3 1e-2 1e-1 \
                    --device gpu \
                    --seed ${SEED} \
                    --exp_name noise-seed
