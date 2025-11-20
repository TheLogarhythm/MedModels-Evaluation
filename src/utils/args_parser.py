import argparse

def parse_training_args():
    """Parse command line arguments for training"""
    parser = argparse.ArgumentParser(description='Train medical image classification models')
    
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset to use')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['medmamba', 'medvit', 'medvitv2'],
                        help='Model architecture to use')
    parser.add_argument('--model_size', type=str, required=True,
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Model size variant')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--learning_rate', type=float, required=True,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use for training')
    
    # Optimizer arguments (for different models)
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam beta1 parameter')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2 parameter')
    parser.add_argument('--lr_decay', type=float, default=None,
                        help='Learning rate decay factor')
    parser.add_argument('--lr_decay_epochs', type=str, default=None,
                        help='Epochs at which to decay learning rate (comma-separated)')
    
    # Logging arguments
    parser.add_argument('--wandb_project', type=str, default='GSCO_baseline',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, 
                        default='royalty-hong-kong-university-of-science-and-technology',
                        help='Weights & Biases entity name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases run name')
    
    return parser.parse_args()