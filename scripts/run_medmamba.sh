#!/bin/bash
set -e

# Source configuration files
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config/datasets.conf"
source "${SCRIPT_DIR}/config/models.conf"
source "${SCRIPT_DIR}/config/hyperparameters.conf"
source "${SCRIPT_DIR}/utils.sh"

# Login to wandb
wandb_login

MODEL="medmamba"
ENV="${MODEL_ENVS[$MODEL]}"
DEVICE="${MODEL_DEVICES[$MODEL]}"

# Train on each dataset
for dataset_name in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Starting training on dataset: $dataset_name"
    echo "=========================================="

    for size in ${MODEL_SIZES[$MODEL]}; do
        # Get hyperparameters
        lr=$(get_hyperparam $MODEL $size "lr")
        bs=$(get_hyperparam $MODEL $size "bs")
        wd=$(get_hyperparam $MODEL $size "wd")
        epochs=$(get_hyperparam $MODEL $size "epochs")

        run_training $MODEL $size $ENV $dataset_name $bs $DEVICE $lr $wd $epochs
    done

    echo "Completed training on dataset: $dataset_name"
    echo ""
done

echo "All MedMamba training completed!"