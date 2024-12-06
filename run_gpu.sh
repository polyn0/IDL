#!/bin/bash

# models to run
# MODEL_NAMES="T0_3B gpt_1.3B"
MODEL_NAMES="llama2_7B"



accelerate launch inference_gpu.py \
    --model_name_list $MODEL_NAMES \
    --sample_n 20 \
    --debug True \
    --batch 1

python bias_score.py \
    --model_name_list $MODEL_NAMES
