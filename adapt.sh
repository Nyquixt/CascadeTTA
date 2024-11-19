#!/bin/bash

PREFIX="--save_path results/ --dataroot data/ --outf checkpoints --cuda --grad_step_online 1 --n_meta 1"

CUDA_VISIBLE_DEVICES=0 python3 main.py $PREFIX --model meta_tta --online_lr 0.001 --nesterov \
    --adapt_batch_size 32 --pretrained 069c8695_pretrained.pth