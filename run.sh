#!/bin/bash

# Define model list
MODEL_LIST=("llama2_7B_chat" "llama3_8B" "gemini")
MODEL_LIST_STR=$(IFS=" " ; echo "${MODEL_LIST[*]}")

# Define mode list
MODE_LIST=('zero' "cot" "self" 'self-cot' 'rci')


# # Nested for loop to iterate over model list and mode list
for mode in "${MODE_LIST[@]}"
do
    echo $mode

    for model in "${MODEL_LIST[@]}"
    do
        echo "Model: $model, Mode: $mode"

        python inference_test.py \
            --model_name $model \
            --device 0 \
            --mode $mode \
            --output_path ./results/$mode/

            # --debug \
            # --sample_n 10
    done

    python bias_score.py \
        --model_name_list $MODEL_LIST_STR \
        --output_path ./results/$mode/
    
done
