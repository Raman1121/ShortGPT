#!/bin/bash

source /home/rdutt/miniconda3/bin/activate anole

CUDA_VISIBLE_DEVICES=2
model="llama3"
pruned_model_ckpt="/nfs/ukrc_roma_ait/models/Pruned_Models/Pruned_llama3_dataset_sec-data-mini_layers_10"
num_layers_to_prune=10

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python /home/rdutt/ShortGPT/short_gpt/calculate_BI.py \
                                            --batch_size 8 \
                                            --model $model \
                                            --do_model_healing \
                                            --healing_batch_size 1 \
                                            --lr 1e-4 \
                                            --num_layers_to_prune $num_layers_to_prune \
                                            --pruned_model_ckpt $pruned_model_ckpt \
                                            --grad_acc_steps 1 \
                                            --use_bnb