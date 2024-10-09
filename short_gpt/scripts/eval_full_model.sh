#!/bin/bash

# source /home/rdutt/miniconda3/bin/activate anole

# CUDA_VISIBLE_DEVICES=1
# num_layers_to_prune=2
# model="llama3.2_1b"

# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python /home/rdutt/ShortGPT/short_gpt/calculate_BI.py \
# --batch_size 8 \
# --num_layers_to_prune $num_layers_to_prune \
# --model $model

# Model Eval

tasks="mmlu"
num_fewshot=5
CUDA_VISIBLE_DEVICES=0

source /home/rdutt/miniconda3/bin/activate lm_eval
cd /home/rdutt/lm-evaluation-harness

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python lm_eval --model hf \
--model_args pretrained=/home/rdutt/Llama-3.2-1B/ \
--tasks $tasks \
--num_fewshot $num_fewshot