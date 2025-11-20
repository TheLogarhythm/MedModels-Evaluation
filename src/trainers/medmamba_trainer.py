"""MedMamba model trainer"""
import sys
import os
import torch
from torch.optim import AdamW
from torchvision import transforms
from trainers.base_trainer import BaseTrainer

class MedMambaTrainer(BaseTrainer):
    """Trainer for MedMamba models"""
    def __init__(self, args, dataset_loader):
        super().__init__(args, dataset_loader)

    def create_model(self):
        """Create MedMamba model"""
        num_classes = len(self.dataset_loader.classes)

        medmamba_path = os.path.join(os.path.dirname(__file__), '../../../MedMamba')
        
        if medmamba_path not in sys.path:
            sys.path.append(medmamba_path)
        
        try:
            from MedMamba import VSSM
            
            if self.args.model_size == 'tiny':
                model = VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=num_classes)
            elif self.args.model_size == 'small':
                model = VSSM(depths=[2, 2, 8, 2], dims=[96, 192, 384, 768], num_classes=num_classes)
            elif self.args.model_size == 'base':
                model = VSSM(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=num_classes)
            else:
                raise ValueError(f"Unsupported model size for MedMamba: {self.args.model_size}")
            
            print(f"✅ Loaded MedMamba-{self.args.model_size} with {num_classes} classes")
            return model
            
        except ImportError as e:
            print(f"❌ Failed to import MedMamba: {e}")
            print(f"📁 MedMamba path: {medmamba_path}")
            raise ImportError("MedMamba not found. Please check the MedMamba path")
    
    def create_optimizer(self):
        """Create AdamW optimizer with custom betas"""
        return AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay
        )
    
    def create_scheduler(self):
        """MedMamba doesn't use learning rate scheduler"""
        return None
    
    def get_transforms(self):
        """Get MedMamba transforms - using MedViT normalization for consistency"""
        MEDMAMBA_MEAN = [0.5, 0.5, 0.5]
        MEDMAMBA_STD = [0.5, 0.5, 0.5]

        return {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(lambda image: image.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEDMAMBA_MEAN, std=MEDMAMBA_STD)
            ]),
            "test": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda image: image.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEDMAMBA_MEAN, std=MEDMAMBA_STD)
            ])
        }