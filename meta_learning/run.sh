##!/bin/bash
m=0
time=$(date "+%Y%m%d-%H%M%S")
for i in {0..4}; do
      LOG_DIR=logs/$meta_learning_seed${seed}.out
      CUDA_VISIBLE_DEVICES=$(($m%4)) python main.py  --save_direct logs --seed $i >$LOG_DIR   &
      m=$(($m+1))
done
wait

