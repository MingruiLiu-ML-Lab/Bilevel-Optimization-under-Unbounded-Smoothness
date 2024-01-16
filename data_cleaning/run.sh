##!/bin/bash
m=0
time=$(date "+%Y%m%d-%H%M%S")
for k in {0..4}; do
    LOG_DIR=logs/logs/$log.out
    CUDA_VISIBLE_DEVICES=$(($m%4)) python main.py  --save_direct logs --seed $k --noise_rate 0.1 >$LOG_DIR &
    m=$(($m+1))
done
wait

