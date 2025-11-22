"""MedViTV2 model trainer"""
import sys
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from trainers.base_trainer import BaseTrainer

class MedViTV2Trainer(BaseTrainer):
    """Trainer for MedViTV2 models"""
    def __init__(self, args, dataset_loader):
        super().__init__(args, dataset_loader)

    def create_model(self):
        """Create MedViTV2 model"""
        num_classes = len(self.dataset_loader.classes)

        medvitv2_path = os.path.join(os.path.dirname(__file__), '../../../MedViTV2')
        
        if medvitv2_path not in sys.path:
            sys.path.append(medvitv2_path)
        
        try:
            from MedViT import MedViT_small, MedViT_base, MedViT_large
            
            if self.args.model_size == 'small':
                model = MedViT_small(pretrained=True, num_classes=num_classes)
            elif self.args.model_size == 'base':
                model = MedViT_base(pretrained=True, num_classes=num_classes)
            elif self.args.model_size == 'large':
                model = MedViT_large(pretrained=True, num_classes=num_classes)
            else:
                raise ValueError(f"Unsupported model size for MedViTV2: {self.args.model_size}")
            
            print(f"✅ Loaded MedViTV2-{self.args.model_size} with {num_classes} classes")
            return model
            
        except ImportError as e:
            print(f"❌ Failed to import MedViTV2: {e}")
            print(f"📁 MedViTV2 path: {medvitv2_path}")
            raise ImportError("MedViTV2 not found. Please check the MedViTV2 path")
    
    def create_optimizer(self):
        """Create AdamW optimizer"""
        return AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay
        )
    
    def create_scheduler(self):
        """Create MultiStepLR scheduler (No scheduler used in original MedViTV2 papers)"""
        # if self.args.lr_decay and self.args.lr_decay_epochs:
        #     milestones = [int(e) for e in self.args.lr_decay_epochs.split(',')]
        #     return MultiStepLR(
        #         self.optimizer,
        #         milestones=milestones,
        #         gamma=self.args.lr_decay
        #     )
        return None
    
    def get_transforms(self):
        """Get MedViTV2 transforms - using MedViT normalization"""
        from utils.config import MEDIMAGE_MEAN, MEDIMAGE_STD
        return {
            "train": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda image: image.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEDIMAGE_MEAN, std=MEDIMAGE_STD)
            ]),
            "test": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda image: image.convert('RGB')),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEDIMAGE_MEAN, std=MEDIMAGE_STD)
            ])
        }