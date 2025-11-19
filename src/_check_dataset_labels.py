#!/usr/bin/env python3
"""Check unique labels and image counts in dataset JSONL files"""

import json
import os
from pathlib import Path
from collections import Counter

def load_jsonl(file_path: str) -> list:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_dataset(dataset_name: str, meta_path: str, data_path: str):
    """Analyze labels and images for a dataset"""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {dataset_name}")
    print(f"{'='*60}")
    
    # Load metadata
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    meta_key = f"D{dataset_name}"
    if meta_key not in metadata:
        print(f"ERROR: {meta_key} not in metadata")
        return
    
    expected_train = metadata[meta_key].get('train_num', 0)
    expected_test = metadata[meta_key].get('test_num', 0)
    label_set = metadata[meta_key].get('label_set', [])
    
    print(f"Expected Train: {expected_train}")
    print(f"Expected Test:  {expected_test}")
    print(f"Label Set:      {label_set}")
    
    # Load JSONL files
    train_meta_path = Path(data_path) / metadata[meta_key]['train_meta']
    test_meta_path = Path(data_path) / metadata[meta_key]['test_meta']
    
    if not train_meta_path.exists():
        print(f"ERROR: Train meta not found: {train_meta_path}")
        return
    if not test_meta_path.exists():
        print(f"ERROR: Test meta not found: {test_meta_path}")
        return
    
    train_data = load_jsonl(str(train_meta_path))
    test_data = load_jsonl(str(test_meta_path))
    
    print(f"Actual Train Lines: {len(train_data)}")
    print(f"Actual Test Lines:  {len(test_data)}")
    
    # Extract labels and images
    train_labels = []
    train_images = []
    for item in train_data:
        label = item.get('label_text')
        if label:
            train_labels.append(label)
        image = item.get('image')
        if image:
            train_images.append(image)
    
    test_labels = []
    test_images = []
    for item in test_data:
        label = item.get('label_text')
        if label:
            test_labels.append(label)
        image = item.get('image')
        if image:
            test_images.append(image)
    
    # Unique labels
    unique_train_labels = set(train_labels)
    unique_test_labels = set(test_labels)
    all_labels = unique_train_labels | unique_test_labels
    
    print(f"\nUnique Train Labels ({len(unique_train_labels)}): {sorted(unique_train_labels)}")
    print(f"Unique Test Labels  ({len(unique_test_labels)}): {sorted(unique_test_labels)}")
    print(f"All Unique Labels   ({len(all_labels)}): {sorted(all_labels)}")
    
    # Check label mismatches
    mismatched_labels = all_labels - set(label_set)
    if mismatched_labels:
        print(f"\n⚠️  Labels not in label_set: {sorted(mismatched_labels)}")
    else:
        print("\n✓ All labels match label_set")
    
    # Image counts and duplicates
    train_image_counts = Counter(train_images)
    test_image_counts = Counter(test_images)
    
    duplicate_train_images = [img for img, count in train_image_counts.items() if count > 1]
    duplicate_test_images = [img for img, count in test_image_counts.items() if count > 1]
    
    print(f"\nTrain Images: {len(train_images)} total, {len(set(train_images))} unique")
    if duplicate_train_images:
        print(f"⚠️  Duplicate Train Images: {len(duplicate_train_images)} (e.g., {duplicate_train_images[:3]})")
    
    print(f"Test Images:  {len(test_images)} total, {len(set(test_images))} unique")
    if duplicate_test_images:
        print(f"⚠️  Duplicate Test Images: {len(duplicate_test_images)} (e.g., {duplicate_test_images[:3]})")
    
    # Summary
    issues = []
    if len(train_data) != expected_train:
        issues.append(f"Train count mismatch: {len(train_data)} vs {expected_train}")
    if len(test_data) != expected_test:
        issues.append(f"Test count mismatch: {len(test_data)} vs {expected_test}")
    if mismatched_labels:
        issues.append(f"Label mismatches: {mismatched_labels}")
    if duplicate_train_images or duplicate_test_images:
        issues.append("Duplicate images found")
    
    if issues:
        print(f"\n❌ Issues: {issues}")
    else:
        print("\n✅ No issues found")

def main():
    meta_path = "/home/sunanhe/luoyi/model_eval/dataset_meta_cls.json"
    data_path = "/jhcnas4/Generalist/shebd/Generalist_Meta_data"
    
    datasets_to_check = ["016_CBIS_DDSM_CALC", "016_CBIS_DDSM_MASS", "075_PCam200", "137_OCT2017"]
    
    for dataset in datasets_to_check:
        analyze_dataset(dataset, meta_path, data_path)

if __name__ == "__main__":
    main()