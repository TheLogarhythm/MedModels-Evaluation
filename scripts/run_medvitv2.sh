#!/bin/bash
set -e

# Source configuration files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/datasets.conf"
source "${SCRIPT_DIR}/config/models.conf"
source "${SCRIPT_DIR}/config/hyperparameters.conf"
source "${SCRIPT_DIR}/config/wandb.conf" 
source "${SCRIPT_DIR}/utils.sh"

# Login to wandb
wandb_login

MODEL="medvitv2"
ENV="${MODEL_ENVS[$MODEL]}"
DEVICE="${MODEL_DEVICES[$MODEL]}"

# Train on each dataset
for dataset_name in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Starting training on dataset: $dataset_name"
    echo "=========================================="

    for size in ${MODEL_SIZES[$MODEL]}; do
        # Get hyperparameters
        lr=$(get_hyperparam $MODEL $size "LR")
        bs=$(get_hyperparam $MODEL $size "BS") 
        wd=$(get_hyperparam $MODEL $size "WD")
        epochs=$(get_hyperparam $MODEL $size "EPOCHS")
        beta1=$(get_hyperparam $MODEL $size "BETA1")
        beta2=$(get_hyperparam $MODEL $size "BETA2")

        run_training $MODEL $size $ENV $dataset_name $bs $DEVICE $lr $wd $epochs $beta1 $beta2
    done

    echo "Completed training on dataset: $dataset_name"
    echo ""
done

echo "All MedViTV2 training completed!"