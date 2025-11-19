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
    train_labels = [item.get('label_text') for item in train_data if item.get('label_text')]
    train_images = [item.get('image') for item in train_data if item.get('image')]
    test_labels = [item.get('label_text') for item in test_data if item.get('label_text')]
    test_images = [item.get('image') for item in test_data if item.get('image')]
    
    # Unique labels
    unique_train_labels = set(train_labels)
    unique_test_labels = set(test_labels)
    all_labels = unique_train_labels | unique_test_labels
    
    print(f"\nUnique Train Labels ({len(unique_train_labels)}): {sorted(unique_train_labels)}")
    print(f"Unique Test Labels  ({len(unique_test_labels)}): {sorted(unique_test_labels)}")
    print(f"All Unique Labels   ({len(all_labels)}): {sorted(all_labels)}")
    
    # Calculate unique images per class
    train_unique_by_class = {}
    for item in train_data:
        label = item.get('label_text')
        image = item.get('image')
        if label and image:
            if label not in train_unique_by_class:
                train_unique_by_class[label] = set()
            train_unique_by_class[label].add(image)
    
    test_unique_by_class = {}
    for item in test_data:
        label = item.get('label_text')
        image = item.get('image')
        if label and image:
            if label not in test_unique_by_class:
                test_unique_by_class[label] = set()
            test_unique_by_class[label].add(image)
    
    # CLASS DISTRIBUTION
    print(f"\n📊 CLASS DISTRIBUTION")
    print(f"{'-'*40}")
    
    # Train set
    train_label_counts = Counter(train_labels)
    total_train = len(train_labels)
    total_unique_train = len(set(train_images))
    
    print(f"\nTRAIN SET:")
    print(f"{'Class':<25} {'Entries':<10} {'Unique':<10} {'%':<8}")
    print(f"{'-'*53}")
    for label in sorted(train_label_counts.keys()):
        entries = train_label_counts[label]
        unique = len(train_unique_by_class.get(label, set()))
        percentage = (entries / total_train) * 100
        print(f"{label:<25} {entries:<10} {unique:<10} {percentage:.1f}%")
    print(f"{'-'*53}")
    print(f"{'TOTAL':<25} {total_train:<10} {total_unique_train:<10} {100.0:.1f}%")
    
    # Test set
    test_label_counts = Counter(test_labels)
    total_test = len(test_labels)
    total_unique_test = len(set(test_images))
    
    print(f"\nTEST SET:")
    print(f"{'Class':<25} {'Entries':<10} {'Unique':<10} {'%':<8}")
    print(f"{'-'*53}")
    for label in sorted(test_label_counts.keys()):
        entries = test_label_counts[label]
        unique = len(test_unique_by_class.get(label, set()))
        percentage = (entries / total_test) * 100
        print(f"{label:<25} {entries:<10} {unique:<10} {percentage:.1f}%")
    print(f"{'-'*53}")
    print(f"{'TOTAL':<25} {total_test:<10} {total_unique_test:<10} {100.0:.1f}%")
    
    # Combined unique images (no overlap between train/test)
    total_unique_images = total_unique_train + total_unique_test
    
    # Check label mismatches
    mismatched_labels = all_labels - set(label_set)
    if mismatched_labels:
        print(f"\n⚠️  Labels not in label_set: {sorted(mismatched_labels)}")
    else:
        print("\n✓ All labels match label_set")
    
    # DUPLICATION ANALYSIS (simplified)
    train_duplicates = len(train_images) - total_unique_train
    test_duplicates = len(test_images) - total_unique_test
    
    print(f"\n📷 DUPLICATION SUMMARY")
    print(f"{'-'*25}")
    print(f"Train: {train_duplicates} duplicate entries ({train_duplicates/len(train_images)*100:.1f}%)")
    print(f"Test:  {test_duplicates} duplicate entries ({test_duplicates/len(test_images)*100:.1f}%)")
    print(f"Total: {train_duplicates + test_duplicates} duplicate entries")
    
    # SUMMARY STATISTICS (FIXED)
    print(f"\n📈 SUMMARY")
    print(f"{'-'*25}")
    print(f"Total entries:    {len(train_data) + len(test_data)}")
    print(f"Unique images:    {total_unique_images}")
    print(f"Duplication rate: {(1 - total_unique_images / (len(train_images) + len(test_images))) * 100:.1f}%")
    
    # Issues
    issues = []
    if len(train_data) != expected_train:
        issues.append(f"Train count mismatch: {len(train_data)} vs {expected_train}")
    if len(test_data) != expected_test:
        issues.append(f"Test count mismatch: {len(test_data)} vs {expected_test}")
    if mismatched_labels:
        issues.append(f"Label mismatches: {mismatched_labels}")
    if train_duplicates > 0 or test_duplicates > 0:
        issues.append(f"Duplicate entries: {train_duplicates} train, {test_duplicates} test")
    
    if issues:
        print(f"\n❌ Issues: {issues}")
    else:
        print("\n✅ No issues found")

def main():
    meta_path = "/home/sunanhe/luoyi/model_eval/dataset_meta_cls.json"
    data_path = "/jhcnas4/Generalist/shebd/Generalist_Meta_data"
    
    
    datasets_to_check = ["016_CBIS_DDSM_CALC"]
    
    for dataset in datasets_to_check:
        analyze_dataset(dataset, meta_path, data_path)

if __name__ == "__main__":
    main()