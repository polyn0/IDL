#!/bin/bash

# models to run
# MODEL_NAMES="gemini"
# MODEL_NAMES="llama3_8B"
MODEL_NAMES="llama2_7B_chat"




python inference.py \
    --model_name_list $MODEL_NAMES \
    --device 1
    # --sample_n 100


python bias_score.py \
    --model_name_list $MODEL_NAMES
