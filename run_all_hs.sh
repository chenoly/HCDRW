#!/bin/bash

# Run all jobs in parallel
nohup python train_hs.py --lambda_stego 1 --lambda_penalty 1 --lambda_z 0 --penalty_start_epoch 10000 --num_epoch 20000 --interval_epoch 5 --prior Laplace --block_mode ResBlock --increase_penalty 1 --gpu_id 0 --batch_size 36 > hs_laplace.log 2>&1 &

nohup python train_hs.py --lambda_stego 1 --lambda_penalty 1 --lambda_z 0.1 --penalty_start_epoch 10000 --num_epoch 20000 --interval_epoch 5 --prior Laplace --block_mode ResBlock --increase_penalty 1 --gpu_id 1 --batch_size 36 > hs_laplace.log 2>&1 &
# Wait for all background jobs to complete
wait

echo "All training jobs finished!"