#!/bin/bash

# Run all jobs in parallel
nohup python3 train_de.py --bpp 1 --lambda_stego 1 --lambda_penalty 1 --prior Laplace --penalty_loss_type MSE --lambda_z 0 --penalty_start_epoch 2000 --num_epoch 10000 --interval_epoch 25 --block_mode ResBlock --increase_penalty 1 --gpu_id 2 --batch_size 36 --train_name NonePrior --continue_train 1 > bpp_1.log 2>&1 &
nohup python3 train_de.py --bpp 2 --lambda_stego 1 --lambda_penalty 1 --prior Laplace --penalty_loss_type MSE --lambda_z 0 --penalty_start_epoch 2000 --num_epoch 10000 --interval_epoch 25 --block_mode ResBlock --increase_penalty 1 --gpu_id 3 --batch_size 36 --train_name NonePrior --continue_train 1 > bpp_2.log 2>&1 &
nohup python3 train_de.py --bpp 3 --lambda_stego 1 --lambda_penalty 1 --prior Laplace --penalty_loss_type MSE --lambda_z 0 --penalty_start_epoch 2000 --num_epoch 10000 --interval_epoch 25 --block_mode ResBlock --increase_penalty 1 --gpu_id 4 --batch_size 36 --train_name NonePrior --continue_train 1 > bpp_3.log 2>&1 &
nohup python3 train_de.py --bpp 1 --lambda_stego 1 --lambda_penalty 1 --prior Laplace --penalty_loss_type MSE --lambda_z 0.1 --penalty_start_epoch 3000 --num_epoch 10000 --interval_epoch 25 --block_mode ResBlock --increase_penalty 1 --gpu_id 5 --batch_size 36 --train_name Prior --continue_train 1 > bpp_1_prior.log 2>&1 &
nohup python3 train_de.py --bpp 2 --lambda_stego 1 --lambda_penalty 1 --prior Laplace --penalty_loss_type MSE --lambda_z 0.1 --penalty_start_epoch 3000 --num_epoch 10000 --interval_epoch 25 --block_mode ResBlock --increase_penalty 1 --gpu_id 6 --batch_size 36 --train_name Prior --continue_train 1 > bpp_2_prior.log 2>&1 &
nohup python3 train_de.py --bpp 3 --lambda_stego 1 --lambda_penalty 1 --prior Laplace --penalty_loss_type MSE --lambda_z 0.1 --penalty_start_epoch 3000 --num_epoch 10000 --interval_epoch 25 --block_mode ResBlock --increase_penalty 1 --gpu_id 7 --batch_size 36 --train_name Prior --continue_train 1 > bpp_3_prior.log 2>&1 &

# Wait for all background jobs to complete
wait

echo "All training jobs finished!"