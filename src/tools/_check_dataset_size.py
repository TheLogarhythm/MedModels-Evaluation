#!/usr/bin/env python3
"""Check expected vs actual dataset sizes for specific datasets"""

import json
import os
from pathlib import Path
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from core.dataset_processor import DatasetProcessor

def main():
    # Datasets you care about
    cared_datasets = [
        "010_RSNA",
        "011_SIIM-ACR",
        "016_CBIS_DDSM_CALC",
        "016_CBIS_DDSM_MASS",
        "017_MedFMC_ENDO",
        "017_MedFMC_COLON",
        "021_PADUFES20",
        "025_Dermnet",
        "027_NLMTB",
        "034_ODIR",
        "037_WCE",
        "043_UBIBC",
        "044_BUSI",
        "046_BUSBRA",
        "051_Derm7PT_Derm",
        "051_Derm7PT_Clinic",
        "075_PCam200",
        "077_RetOCT",
        "136_ChestXRay2017",
        "137_OCT2017"
    ]
    
    # Load metadata
    meta_path = "/home/sunanhe/luoyi/model_eval/dataset_meta_cls.json"
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    # Initialize processor to get actual sizes
    processor = DatasetProcessor(cared_datasets)
    
    print(f"{'='*80}")
    print("DATASET SIZE CHECK (CARED DATASETS)")
    print(f"{'='*80}")
    print(f"{'Dataset':<20} {'Expected Train':<15} {'Actual Train':<15} {'Expected Test':<15} {'Actual Test':<15} {'Status'}")
    print(f"{'-'*80}")
    
    for dataset_name in cared_datasets:
        dataset_key = f"D{dataset_name}"
        
        if dataset_key not in metadata:
            print(f"{dataset_name:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'✗ No metadata'}")
            continue
        
        info = metadata[dataset_key]
        expected_train = info.get('train_num', 0)
        expected_test = info.get('test_num', 0)
        
        actual_train, actual_test = processor._get_local_dataset_size(dataset_name)

        if actual_train == expected_train and actual_test == expected_test:
            status = "✓ Prepared"
        elif actual_train > 0 or actual_test > 0:
            status = "⚠️ Partial"
        else:
            status = "✗ Not prepared"
        
        print(f"{dataset_name:<20} {expected_train:<15} {actual_train:<15} {expected_test:<15} {actual_test:<15} {status}")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()