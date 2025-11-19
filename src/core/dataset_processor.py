import os
import json
import shutil
from typing import Dict, List, Tuple, Optional
import re

class DatasetProcessor:
    """Creates local train/test folder structure"""
    
    def __init__(self,
                 selected_datasets: List[str],
                 dataset_meta_path: str = "/home/sunanhe/luoyi/model_eval/dataset_meta_cls.json",
                 meta_data_base_path: str = "/jhcnas4/Generalist/shebd/Generalist_Meta_data",
                 dataset_base_path: str = "/jhcnas4/Generalist/Medical_Data_2025",
                 local_dataset_base_path: str = "/home/sunanhe/luoyi/model_eval/datasets"):
        
        self.selected_datasets = selected_datasets
        self.dataset_meta_path = dataset_meta_path
        self.meta_data_base_path = meta_data_base_path
        self.dataset_base_path = dataset_base_path
        self.local_dataset_base_path = local_dataset_base_path
        self.dataset_meta = self._load_json_file(self.dataset_meta_path)
    
    def _load_json_file(self, path) -> Dict:
        """Load dataset metadata from JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _load_jsonl_file(self, path) -> List[Dict]:
        """Load train/test set metadata from JSONL file."""
        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def _get_local_dataset_size(self, dataset_name: str) -> Tuple[int, int]:
        """Get the number of training and testing samples for a dataset."""
        local_train_path = os.path.join(self.local_dataset_base_path, dataset_name, 'train')
        local_test_path = os.path.join(self.local_dataset_base_path, dataset_name, 'test')
        
        local_train_num = 0
        local_test_num = 0
        
        if os.path.exists(local_train_path):
            for root, dirs, files in os.walk(local_train_path):
                local_train_num += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        
        if os.path.exists(local_test_path):
            for root, dirs, files in os.walk(local_test_path):
                local_test_num += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
            
        return local_train_num, local_test_num

    def map_label(self, dataset_name: str, label_text: str) -> str:
        """Map mismatched labels to correct ones for specific datasets.
        
        Args:
            dataset_name: Name of the dataset (e.g., '010_RSNA')
            label_text: Original label text from JSONL
            
        Returns:
            Mapped label text
        """
        # Define mappings for known mismatches
        mappings = {
            "010_RSNA": {
                "exact": {
                    "No Finding": "normal",
                    "Pneumonia": "pneumonia"
                },
                "regex": {}
            },
            "016_CBIS_DDSM_CALC": {
                "exact": {
                    "benign_without_callback": "benign"
                },
                "regex": {}
            },
            "016_CBIS_DDSM_MASS": {
                "exact": {
                    "benign_without_callback": "benign"
                },
                "regex": {}
            },
            "044_BUSI": {
                "exact": {
                    "normal": "benign"
                },
                "regex": {}
            },
            "051_Derm7PT_Derm": {
                "exact": {},
                "regex": {
                    r".*nevus.*": "nevus"  # any label containing "nevus"
                }
            },
            "051_Derm7PT_Clinic": {
                "exact": {},
                "regex": {
                    r".*nevus.*": "nevus"
                }
            },
            "137_OCT2017": {
                "exact": {
                    "cnv": "choroidal neovascularization",
                    "dme": "diabetic macular edema"
                },
                "regex": {}
            }
            # Add more datasets as needed
        }
        
        # Get mappings for this dataset
        dataset_mappings = mappings.get(dataset_name, {"exact": {}, "regex": {}})
        
        # Check exact matches first
        if label_text in dataset_mappings["exact"]:
            return dataset_mappings["exact"][label_text]
        
        # Check regex patterns
        for pattern, mapped_label in dataset_mappings["regex"].items():
            if re.search(pattern, label_text, re.IGNORECASE):
                return mapped_label
        
        # Return original if no match
        return label_text

    def check_dataset_exists_locally(self, dataset_name: str) -> bool:
        """Check if the dataset exists in the local directory."""
        local_train_num, local_test_num = self._get_local_dataset_size(dataset_name)
        
        # Special handling for datasets with duplicate images in JSONL
        duplicate_datasets = {"016_CBIS_DDSM_CALC", "016_CBIS_DDSM_MASS"}
        if dataset_name in duplicate_datasets:
            # Use unique image count from JSONL instead of total lines
            train_meta = self._load_jsonl_file(os.path.join(self.meta_data_base_path, self.dataset_meta['D'+dataset_name]['train_meta']))
            test_meta = self._load_jsonl_file(os.path.join(self.meta_data_base_path, self.dataset_meta['D'+dataset_name]['test_meta']))
            
            unique_train_images = len(set(item.get('image') for item in train_meta if item.get('image')))
            unique_test_images = len(set(item.get('image') for item in test_meta if item.get('image')))
            
            meta_train_num = unique_train_images
            meta_test_num = unique_test_images
        else:
            meta_train_num = self.dataset_meta['D'+dataset_name]['train_num']
            meta_test_num = self.dataset_meta['D'+dataset_name]['test_num']
        
        return local_train_num == meta_train_num and local_test_num == meta_test_num
    
    def copy_dataset_to_local(self, dataset_name: str) -> None:
        """Copy dataset from the base directory to the local directory."""
        if self.check_dataset_exists_locally(dataset_name):
            print(f"Dataset {dataset_name} already exists locally.")
            return

        src_path = os.path.join(self.dataset_base_path, dataset_name)
        dst_path = os.path.join(self.local_dataset_base_path, dataset_name)
        
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)

        label_set = self.dataset_meta['D'+dataset_name]['label_set']
        train_meta = self._load_jsonl_file(os.path.join(self.meta_data_base_path, self.dataset_meta['D'+dataset_name]['train_meta']))
        test_meta = self._load_jsonl_file(os.path.join(self.meta_data_base_path, self.dataset_meta['D'+dataset_name]['test_meta']))
        
        os.makedirs(dst_path, exist_ok=True)
        train_path = os.path.join(dst_path, 'train')
        test_path = os.path.join(dst_path, 'test')
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        if self.dataset_meta['D'+dataset_name]['tasktype'] == 'multilabel':
            for line in train_meta:
                src_image_path = os.path.join(self.dataset_base_path, line['image'])
                dst_image_directory = os.path.join(self.local_dataset_base_path, dataset_name, 'train')
                shutil.copy(src_image_path, dst_image_directory)
            
            for line in test_meta:
                src_image_path = os.path.join(self.dataset_base_path, line['image'])
                dst_image_directory = os.path.join(self.local_dataset_base_path, dataset_name, 'test')
                shutil.copy(src_image_path, dst_image_directory)
        else:
            for line in train_meta:
                image = line['image']
                label_text = line['label_text'].strip()
                label = self.map_label(dataset_name, label_text)

                if label not in label_set:
                    raise ValueError(f"Label '{label}' not found in label set for dataset '{dataset_name}'.")

                src_image_path = os.path.join(self.dataset_base_path, image)
                dst_image_directory = os.path.join(self.local_dataset_base_path, dataset_name, 'train', label)
                os.makedirs(dst_image_directory, exist_ok=True)
                shutil.copy(src_image_path, dst_image_directory)

            for line in test_meta:
                image = line['image']
                label_text = line['label_text'].strip()
                label = self.map_label(dataset_name, label_text)

                if label not in label_set:
                    raise ValueError(f"Label '{label}' not found in label set for dataset '{dataset_name}'.")

                src_image_path = os.path.join(self.dataset_base_path, image)
                dst_image_directory = os.path.join(self.local_dataset_base_path, dataset_name, 'test', label)
                os.makedirs(dst_image_directory, exist_ok=True)
                shutil.copy(src_image_path, dst_image_directory)