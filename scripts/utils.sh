#!/bin/bash

# Common utility functions
WANDB_API_KEY="949480fc468133a90171487e61c9830a949d9872"
WANDB_ENTITY="royalty-hong-kong-university-of-science-and-technology"
SRC_DIR="/home/sunanhe/luoyi/model_eval/src"

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
    
    echo "Training $model-$size on $dataset_name..."
    echo "  LR: $learning_rate, BS: $batch_size, WD: $weight_decay, Epochs: $epochs"

    # Get additional hyperparameters if they exist
    local lr_decay=$(get_hyperparam $model $size "lr_decay")
    local lr_decay_epochs=$(get_hyperparam $model $size "lr_decay_epochs")
    local beta1=$(get_hyperparam $model $size "beta1")
    local beta2=$(get_hyperparam $model $size "beta2")

    # Build command with optional parameters
    local cmd="conda run -n $env python ${SRC_DIR}/train_${model}.py \
        --dataset_name $dataset_name \
        --model_name $model \
        --model_size $size \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project medical-image-classification \
        --wandb_name ${model}_${size}_${dataset_name}_$(date +%Y%m%d_%H%M%S) \
        --batch_size $batch_size \
        --device $device \
        --learning_rate $learning_rate \
        --weight_decay $weight_decay \
        --epochs $epochs"

    # Add optional parameters if they exist
    if [ ! -z "$lr_decay" ]; then
        cmd="$cmd --lr_decay $lr_decay"
    fi
    if [ ! -z "$lr_decay_epochs" ]; then
        cmd="$cmd --lr_decay_epochs $lr_decay_epochs"
    fi
    if [ ! -z "$beta1" ]; then
        cmd="$cmd --beta1 $beta1"
    fi
    if [ ! -z "$beta2" ]; then
        cmd="$cmd --beta2 $beta2"
    fi

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