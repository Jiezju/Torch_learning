#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1,2,3,4 \
    python -m torch.distributed.launch \
           --nproc_per_node=4 main.py \
           -a mobilenet_v2 -b 160 \

CUDA_VISIBLE_DEVICES=1,2,3,4 \
    python -m torch.distributed.launch \
           --nproc_per_node=4 main_fp16.py \
           -a mobilenet_v2 -b 320 \
           --opt-level O1 --loss-scale 128.0
