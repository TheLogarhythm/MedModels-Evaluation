# Medical Image Classification Benchmark

A comprehensive benchmarking framework for medical image classification models including MedViT, MedViTv2, and MedMamba.

## Project Structure

```
MedModels-Evaluation/
├── scripts/                     # Training and evaluation scripts
│   ├── run_medvit.sh            # Train MedViT models on all datasets
│   ├── run_medvitv2.sh          # Train MedViTv2 models on all datasets  
│   ├── run_medmamba.sh          # Train MedMamba models on all datasets
│   ├── config/
│   │   ├── datasets.conf        # Dataset configurations and paths
│   │   ├── models.conf          # Model configurations and devices
│   │   └── hyperparameters.conf # Training hyperparameters
│   └── utils.sh                 # Common utility functions
├── src/                         # Source code
│   ├── train.py                 # Unified training script (main entry point)
│   ├── core/
│   │   ├── dataset_processor.py # Dataset preprocessing and validation
│   │   ├── dataset_loader.py    # Data loading and augmentation
│   │   ├── metrics.py           # Evaluation metrics calculation
│   │   └── training_logger.py   # Training progress tracking
│   ├── trainers/
│   │   ├── base_trainer.py      # Base training class
│   │   ├── medvit_trainer.py    # MedViT specific training
│   │   ├── medvitv2_trainer.py  # MedViTv2 specific training
│   │   └── medmamba_trainer.py  # MedMamba specific training
│   └── utils/
│       ├── args_parser.py       # Command line argument parsing
│       └── config.py            # Configuration constants
├── datasets/                    # Local dataset storage
├── eval_results/                # Training results and model checkpoints
└── logs/                        # Training logs and history
```

## Scripts Overview

### Training Scripts
- **`run_medvit.sh`** - Trains MedViT models on all configured datasets
- **`run_medvitv2.sh`** - Trains MedViTv2 models on all configured datasets  
- **`run_medmamba.sh`** - Trains MedMamba models on all configured datasets

### Configuration Files
- **`datasets.conf`** - Defines datasets to evaluate and their properties
- **`models.conf`** - Configures model environments and GPU devices
- **`hyperparameters.conf`** - Sets learning rates, batch sizes, etc. for each model

### Core Components
- **`train.py`** - Main training orchestrator, handles dataset loading and trainer selection
- **`base_trainer.py`** - Implements common training loop, metrics logging, model saving
- **Model-specific trainers** - Handle model initialization, optimizers, and transforms

## Supported Models

- **MedViT** - Medical Vision Transformer
- **MedViTv2** - Enhanced Medical Vision Transformer with AugMix
- **MedMamba** - Medical Mamba-based architecture

## Quick Start

### 1. Setup Environment
```bash
cd /home/sunanhe/luoyi/MedModels-Evaluation

# Activate appropriate conda environment
conda activate medvitv2-ly  # or medmamba, etc.
```

### 2. Run Training
```bash
# Train MedViT on all datasets
./scripts/run_medvit.sh

# Train MedViTv2 on all datasets  
./scripts/run_medvitv2.sh

# Train MedMamba on all datasets
./scripts/run_medmamba.sh
```

### 3. Monitor Training
```bash
# Check active training sessions
tmux ls

# Attach to monitor progress
tmux attach -t medvit

# Detach (keep running in background)
Ctrl + B, then D
```

## Advanced Usage

### Single Dataset Training
```bash
# Modify datasets.conf to include only specific datasets
# Then run the training script as normal
```

### Custom Hyperparameters
Edit `scripts/config/hyperparameters.conf` to modify:
- Learning rates
- Batch sizes  
- Weight decay
- Training epochs
- Optimizer betas

### Resume Training
Delete the corresponding directory in `eval_results/` to retrain:
```bash
rm -rf eval_results/medvit_base_010_RSNA
```
