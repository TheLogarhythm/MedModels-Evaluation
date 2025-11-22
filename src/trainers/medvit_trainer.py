"""MedViT model trainer"""
import sys
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from trainers.base_trainer import BaseTrainer

class MedViTTrainer(BaseTrainer):
    """Trainer for MedViT models"""
    def __init__(self, args, dataset_loader):
        super().__init__(args, dataset_loader)

    def create_model(self):
        """Create MedViT model"""
        num_classes = len(self.dataset_loader.classes)

        medvit_path = os.path.join(os.path.dirname(__file__), '../../../MedViT')
        
        if medvit_path not in sys.path:
            sys.path.append(medvit_path)
        
        try:
            from MedViT import MedViT_small, MedViT_base, MedViT_large
            
            if self.args.model_size == 'small':
                model = MedViT_small(pretrained=True, num_classes=num_classes)
            elif self.args.model_size == 'base':
                model = MedViT_base(pretrained=True, num_classes=num_classes)
            elif self.args.model_size == 'large':
                model = MedViT_large(pretrained=True, num_classes=num_classes)
            else:
                raise ValueError(f"Unsupported model size for MedViT: {self.args.model_size}")
            
            print(f"✅ Loaded MedViT-{self.args.model_size} with {num_classes} classes")
            return model
            
        except ImportError as e:
            print(f"❌ Failed to import MedViT: {e}")
            print(f"📁 MedViT path: {medvit_path}")
            raise ImportError("MedViT not found. Please check the MedViT path")
    
    def create_optimizer(self):
        """Create AdamW optimizer"""
        return AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,  # 0.0001
            betas=(0.9, 0.999),         # B1=0.9, B2=0.999
            weight_decay=self.args.weight_decay  # 1e-4
        )
    
    def create_scheduler(self):
        """Create MultiStepLR scheduler (No scheduler used in original MedViT papers)"""
        # if self.args.lr_decay and self.args.lr_decay_epochs:
        #     milestones = [int(e) for e in self.args.lr_decay_epochs.split(',')]
        #     return MultiStepLR(
        #         self.optimizer,
        #         milestones=milestones,
        #         gamma=self.args.lr_decay
        #     )
        return None
    
    def get_transforms(self):
        """Get MedViT transforms"""
        from utils.config import MEDIMAGE_MEAN, MEDIMAGE_STD

        return {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEDIMAGE_MEAN, std=MEDIMAGE_STD)
            ]),
            "test": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEDIMAGE_MEAN, std=MEDIMAGE_STD)
            ])
        }