#!/bin/bash

TASK="asqp"
DATASET="rest15_top3"
MODEL_NAME_OR_PATH="t5-base"
NUM_REASONINGS=16
BATCH_SIZE=16

for SEED in 42
do

python src/main.py \
            --task $TASK \
            --dataset $DATASET \
            --num_reasonings $NUM_REASONINGS \
            --model_name_or_path $MODEL_NAME_OR_PATH \
            --n_gpu 1 \
            --do_train \
            --do_inference \
            --do_self_consistency \
            --seed $SEED \
            --batch_size $BATCH_SIZE \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 1 \
            --num_return_sequences 15 \

done