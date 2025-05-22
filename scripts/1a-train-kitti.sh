#!/bin/bash

savepath=$1;

python main_2.py --mode=train --dataset="kitti-raw" --seq_len=4 --db_seq_len=8 --arch_depth=6 --ckpt_dir="$savepath" --log_dir="$savepath/summaries" --records=data/kitti-raw-reduced/train_data/ --enable_validation $2
