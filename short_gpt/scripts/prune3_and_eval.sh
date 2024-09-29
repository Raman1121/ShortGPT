#!/bin/bash

source /home/rdutt/miniconda3/bin/activate anole

CUDA_VISIBLE_DEVICES=2
num_layers_to_prune=3
model="llama3.2_3b"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python /home/rdutt/ShortGPT/short_gpt/calculate_BI.py \
--batch_size 8 \
--num_layers_to_prune $num_layers_to_prune \
--model $model

# Model Eval

tasks="mmlu"
num_fewshot=5

source /home/rdutt/miniconda3/bin/activate lm_eval
cd /home/rdutt/lm-evaluation-harness

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python lm_eval --model hf \
--model_args pretrained=/nfs/ukrc_roma_ait/models/Pruned_Models/Pruned_llama3.2_3b_dataset_sec-data-mini_layers_3 \
--tasks $tasks \
--num_fewshot $num_fewshot