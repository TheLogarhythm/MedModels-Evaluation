import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Optional, Callable, Tuple, Dict, List
import os
import json
from PIL import Image

class CustomImageDataset(Dataset):
    """Custom dataset that respects user-defined class_to_idx mapping"""
    
    def __init__(self, root_dir: str, class_to_idx: Dict[str, int], transform=None):
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.classes = list(class_to_idx.keys())
        self.samples = []
        self._build_samples()
    
    def _build_samples(self):
        """Build list of (image_path, class_idx) tuples"""
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"WARNING: Class directory not found: {class_dir}")
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"ERROR: Failed to load image {img_path}: {e}")
            # Return a blank image or skip (for now, raise)
            raise e
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DatasetLoader:
    """Dataset loader for ImageNet-structured dataset with metadata integration"""

    def __init__(self,
                 dataset_name: str,
                 dataset_meta_path: str = "/home/sunanhe/luoyi/model_eval/dataset_meta_cls.json",
                 local_dataset_base_path: str = "/home/sunanhe/luoyi/model_eval/datasets",
                 batch_size: int = 32,
                 num_workers: int = 4):
        
        self.dataset_name = dataset_name
        self.dataset_meta_path = dataset_meta_path
        self.local_dataset_base_path = local_dataset_base_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Load metadata
        with open(dataset_meta_path, 'r') as f:
            metadata = json.load(f)
        
        meta_key = f"D{dataset_name}"
        if meta_key not in metadata:
            raise ValueError(f"Dataset '{dataset_name}' not found in metadata at {dataset_meta_path}")
        
        self.label_set = metadata[meta_key]['label_set']
        self.class_to_idx = {label: idx for idx, label in enumerate(self.label_set)}
        self.classes = self.label_set
        
        self.dataset_path = os.path.join(local_dataset_base_path, dataset_name)
        self.train_path = os.path.join(self.dataset_path, 'train')
        self.test_path = os.path.join(self.dataset_path, 'test')
        
        # Verify paths exist
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Train path not found: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test path not found: {self.test_path}")

    def get_train_loader(self, train_transform) -> DataLoader:
        """Get training data loader"""
        train_set = CustomImageDataset(
            root_dir=self.train_path,
            class_to_idx=self.class_to_idx,
            transform=train_transform
        )
        
        if len(train_set) == 0:
            raise ValueError(f"No training images found in {self.train_path}")
        
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_test_loader(self, test_transform) -> DataLoader:
        """Get testing data loader"""
        test_set = CustomImageDataset(
            root_dir=self.test_path,
            class_to_idx=self.class_to_idx,
            transform=test_transform
        )
        
        if len(test_set) == 0:
            raise ValueError(f"No test images found in {self.test_path}")
        
        return DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_class_names(self) -> list:
        """Get class names from the dataset"""
        return self.classes

    def get_class_to_idx(self) -> dict:
        """Get class to index mapping"""
        return self.class_to_idx
    
    def get_dataset_info(self) -> dict:
        """Get dataset information"""
        return {
            'name': self.dataset_name,
            'classes': self.classes,
            'num_classes': len(self.classes),
            'train_path': self.train_path,
            'test_path': self.test_path
        }
    
    def get_task_type(self) -> str:
        """Get task type from metadata"""
        # Load metadata to get task_type
        with open(self.dataset_meta_path, 'r') as f:
            metadata = json.load(f)
        
        meta_key = f"D{self.dataset_name}"
        
        return metadata[meta_key].get('tasktype')