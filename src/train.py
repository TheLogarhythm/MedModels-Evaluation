"""Unified training script for all models"""
import os
import sys

# Add the project root to path to enable absolute imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.core.dataset_processor import DatasetProcessor
from src.core.dataset_loader import DatasetLoader
from src.trainers.medvit_trainer import MedViTTrainer
from src.trainers.medvitv2_trainer import MedViTV2Trainer
from src.trainers.medmamba_trainer import MedMambaTrainer
from src.utils.args_parser import parse_training_args
from src.utils.config import LOCAL_DATASET_BASE

def main():
    """Main training function"""
    args = parse_training_args()
    
    # Initialize dataset processor
    dataset_processor = DatasetProcessor([args.dataset_name])
    if not dataset_processor.check_dataset_exists_locally(args.dataset_name):
        print(f"Copying dataset {args.dataset_name} to local directory...")
        dataset_processor.copy_dataset_to_local(args.dataset_name)
    
    # Initialize dataset loader
    dataset_loader = DatasetLoader(
        dataset_name=args.dataset_name,
        local_dataset_base_path=LOCAL_DATASET_BASE,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Select trainer based on model
    trainer_map = {
        'medvit': MedViTTrainer,
        'medvitv2': MedViTV2Trainer,
        'medmamba': MedMambaTrainer
    }
    
    if args.model_name not in trainer_map:
        raise ValueError(f"Unknown model: {args.model_name}")
    
    # Initialize and run trainer - FIXED: Only pass 2 arguments
    trainer_class = trainer_map[args.model_name]
    trainer = trainer_class(args, dataset_loader)  # Remove dataset_processor
    trainer.train()

if __name__ == "__main__":
    main()