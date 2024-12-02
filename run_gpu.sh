#!/bin/bash

# models to run
MODEL_NAMES="T0_3B gpt_1.3B"


accelerate launch inference_gpu.py \
    --model_name_list $MODEL_NAMES


python bias_score.py \
    --model_name_list $MODEL_NAMES
