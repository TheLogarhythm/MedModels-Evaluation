#!/bin/bash

# Login to wandb
wandb_login() {
    echo "Logging in to Weights & Biases..."
    wandb login $WANDB_API_KEY
}

# Function to run training with specific model
run_training() {
    local model=$1
    local size=$2
    local env=$3
    local dataset_name=$4
    local batch_size=$5
    local device=$6
    local learning_rate=$7
    local weight_decay=$8
    local epochs=$9
    local beta1=${10} 
    local beta2=${11}

    echo "Training $model-$size on $dataset_name..."
    echo "  LR: $learning_rate, BS: $batch_size, WD: $weight_decay, Epochs: $epochs"

    # Build command - using unified train.py with proper path
    local cmd="conda run -n $env python ${SRC_DIR}/train.py \
        --dataset_name $dataset_name \
        --model_name $model \
        --model_size $size \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project GSCO_baseline \
        --wandb_name ${model}_${size}_${dataset_name}_$(date +%Y%m%d_%H%M%S) \
        --batch_size $batch_size \
        --device $device \
        --learning_rate $learning_rate \
        --weight_decay $weight_decay \
        --epochs $epochs"

    # Add optional parameters if they exist
    if [ ! -z "$beta1" ] && [ "$beta1" != "null" ]; then
        cmd="$cmd --beta1 $beta1"
    fi
    if [ ! -z "$beta2" ] && [ "$beta2" != "null" ]; then
        cmd="$cmd --beta2 $beta2"
    fi

    # Execute the command
    echo "Running: $cmd"
    eval $cmd
}

# Get hyperparameter value
get_hyperparam() {
    local model=$1
    local size=$2
    local param=$3
    
    local var_name="${model^^}_${size^^}_${param^^}"
    echo "${!var_name}"
}