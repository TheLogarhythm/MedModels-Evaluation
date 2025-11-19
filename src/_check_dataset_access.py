"""Check dataset accessibility without copying files"""
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


class DatasetAccessChecker:
    """Check accessibility of all datasets without copying files"""
    
    def __init__(self,
                 dataset_meta_path: str = "/home/sunanhe/luoyi/model_eval/dataset_meta_cls.json",
                 meta_data_base_path: str = "/jhcnas4/Generalist/shebd/Generalist_Meta_data",
                 dataset_base_path: str = "/jhcnas4/Generalist/Medical_Data_2025"):
        
        self.dataset_meta_path = dataset_meta_path
        self.meta_data_base_path = meta_data_base_path
        self.dataset_base_path = dataset_base_path
        self.dataset_meta = self._load_json_file(self.dataset_meta_path)
    
    def _load_json_file(self, path) -> Dict:
        """Load dataset metadata from JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Failed to load metadata file {path}: {e}")
            return {}
    
    def _load_jsonl_file(self, path) -> List[Dict]:
        """Load train/test set metadata from JSONL file."""
        data = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        except Exception as e:
            print(f"❌ Failed to load JSONL file {path}: {e}")
            return []
    
    def check_metadata_access(self, dataset_name: str) -> Tuple[bool, int, int]:
        """Check if we can access dataset metadata files."""
        try:
            meta_key = 'D' + dataset_name
            if meta_key not in self.dataset_meta:
                return False, 0, 0
            
            dataset_info = self.dataset_meta[meta_key]
            
            # Check train metadata file
            train_meta_path = os.path.join(self.meta_data_base_path, dataset_info['train_meta'])
            if not os.path.exists(train_meta_path):
                return False, 0, 0
            
            # Check test metadata file  
            test_meta_path = os.path.join(self.meta_data_base_path, dataset_info['test_meta'])
            if not os.path.exists(test_meta_path):
                return False, 0, 0
            
            # Try to load metadata to verify it's readable
            train_meta = self._load_jsonl_file(train_meta_path)
            test_meta = self._load_jsonl_file(test_meta_path)
            
            train_count = len(train_meta)
            test_count = len(test_meta)
            
            return True, train_count, test_count
            
        except Exception as e:
            print(f"❌ Error checking metadata for {dataset_name}: {e}")
            return False, 0, 0
    
    def check_sample_files_access(self, dataset_name: str, max_samples: int = 5) -> Tuple[bool, int, int]:
        """Check if we can access actual image files by testing a few samples."""
        try:
            meta_key = 'D' + dataset_name
            if meta_key not in self.dataset_meta:
                return False, 0, 0
            
            dataset_info = self.dataset_meta[meta_key]
            
            # Load metadata
            train_meta_path = os.path.join(self.meta_data_base_path, dataset_info['train_meta'])
            test_meta_path = os.path.join(self.meta_data_base_path, dataset_info['test_meta'])
            
            train_meta = self._load_jsonl_file(train_meta_path)
            test_meta = self._load_jsonl_file(test_meta_path)
            
            if not train_meta or not test_meta:
                return False, 0, 0
            
            # Test access to a few training samples
            train_accessible = 0
            for i, sample in enumerate(train_meta[:max_samples]):
                image_path = os.path.join(self.dataset_base_path, sample['image'])
                if os.path.exists(image_path):
                    train_accessible += 1
            
            # Test access to a few test samples
            test_accessible = 0
            for i, sample in enumerate(test_meta[:max_samples]):
                image_path = os.path.join(self.dataset_base_path, sample['image'])
                if os.path.exists(image_path):
                    test_accessible += 1
            
            success = (train_accessible > 0 and test_accessible > 0)
            return success, train_accessible, test_accessible
            
        except Exception as e:
            print(f"❌ Error checking sample files for {dataset_name}: {e}")
            return False, 0, 0
    
    def check_dataset_access(self, dataset_name: str, test_samples: int = 3) -> Dict:
        """Comprehensive access check for a single dataset."""
        print(f"🔍 Checking {dataset_name}...")
        
        start_time = time.time()
        
        # Check metadata access
        meta_accessible, train_count, test_count = self.check_metadata_access(dataset_name)
        
        # Check sample file access
        files_accessible = False
        train_accessible_samples = 0
        test_accessible_samples = 0
        
        if meta_accessible:
            files_accessible, train_accessible_samples, test_accessible_samples = \
                self.check_sample_files_access(dataset_name, test_samples)
        
        duration = time.time() - start_time
        
        result = {
            'dataset': dataset_name,
            'meta_accessible': meta_accessible,
            'files_accessible': files_accessible,
            'train_samples_total': train_count,
            'test_samples_total': test_count,
            'train_samples_checked': train_accessible_samples,
            'test_samples_checked': test_accessible_samples,
            'duration_seconds': round(duration, 2)
        }
        
        # Print quick status
        status_icon = "✅" if (meta_accessible and files_accessible) else "❌"
        print(f"   {status_icon} Metadata: {meta_accessible}, Files: {files_accessible}, "
              f"Train: {train_count}, Test: {test_count}, Time: {duration:.2f}s")
        
        return result
    
    def check_all_datasets(self, dataset_names: List[str], test_samples: int = 3) -> List[Dict]:
        """Check access for all specified datasets."""
        print("🚀 Starting dataset access check...")
        print(f"📁 Metadata path: {self.meta_data_base_path}")
        print(f"📁 Data path: {self.dataset_base_path}")
        print(f"🔢 Testing {test_samples} samples per split")
        print("-" * 80)
        
        results = []
        accessible_count = 0
        
        for dataset_name in dataset_names:
            result = self.check_dataset_access(dataset_name, test_samples)
            results.append(result)
            
            if result['meta_accessible'] and result['files_accessible']:
                accessible_count += 1
        
        return results, accessible_count


def print_summary(results: List[Dict], total_datasets: int):
    """Print a comprehensive summary of access check results."""
    print("\n" + "=" * 80)
    print("📊 DATASET ACCESS CHECK SUMMARY")
    print("=" * 80)
    
    accessible = [r for r in results if r['meta_accessible'] and r['files_accessible']]
    meta_only = [r for r in results if r['meta_accessible'] and not r['files_accessible']]
    inaccessible = [r for r in results if not r['meta_accessible']]
    
    print(f"Total datasets:          {total_datasets}")
    print(f"✅ Fully accessible:     {len(accessible)}")
    print(f"⚠️  Metadata only:        {len(meta_only)}")
    print(f"❌ Completely inaccessible: {len(inaccessible)}")
    
    if accessible:
        print(f"\n✅ FULLY ACCESSIBLE DATASETS:")
        for result in accessible:
            print(f"   • {result['dataset']} "
                  f"(Train: {result['train_samples_total']}, "
                  f"Test: {result['test_samples_total']})")
    
    if meta_only:
        print(f"\n⚠️  METADATA ONLY (NO FILE ACCESS):")
        for result in meta_only:
            print(f"   • {result['dataset']} "
                f"(Train: {result['train_samples_checked']}/{result['train_samples_total']} accessible, "
                f"Test: {result['test_samples_checked']}/{result['test_samples_total']} accessible)")
    
    if inaccessible:
        print(f"\n❌ INACCESSIBLE DATASETS:")
        for result in inaccessible:
            print(f"   • {result['dataset']}")
    
    # Calculate statistics
    total_train = sum(r['train_samples_total'] for r in accessible)
    total_test = sum(r['test_samples_total'] for r in accessible)
    total_samples = total_train + total_test
    
    print(f"\n📈 ACCESSIBLE DATA STATISTICS:")
    print(f"   Total train samples:  {total_train:,}")
    print(f"   Total test samples:   {total_test:,}")
    print(f"   Total samples:        {total_samples:,}")
    
    if inaccessible or meta_only:
        print(f"\n🔧 NEXT STEPS:")
        if meta_only:
            print(f"   • Request file access for {len(meta_only)} datasets with metadata only")
        if inaccessible:
            print(f"   • Request complete access for {len(inaccessible)} inaccessible datasets")
        
        inaccessible_names = [r['dataset'] for r in inaccessible + meta_only]
        print(f"\n   Datasets needing attention: {', '.join(inaccessible_names)}")


def main():
    """Main function to check dataset access."""
    parser = argparse.ArgumentParser(description='Check dataset accessibility without copying files')
    parser.add_argument('--test-samples', type=int, default=3,
                       help='Number of sample files to test per dataset (default: 3)')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to check (default: all available)')
    parser.add_argument('--output', type=str,
                       help='Output JSON file to save results')
    
    args = parser.parse_args()
    
    # Define all available datasets
    all_datasets = [
        "010_RSNA", "011_SIIM-ACR", "016_CBIS_DDSM_CALC", "016_CBIS_DDSM_MASS",
        "017_MedFMC_ENDO", "017_MedFMC_COLON", "021_PADUFES20", "025_Dermnet",
        "027_NLMTB", "034_ODIR", "037_WCE", "043_UBIBC", "044_BUSI", "046_BUSBRA",
        "051_Derm7PT_Derm", "051_Derm7PT_Clinic", "075_PCam200", "077_RetOCT",
        "136_ChestXRay2017", "137_OCT2017"
    ]
    
    # Determine which datasets to check
    datasets_to_check = args.datasets if args.datasets else all_datasets
    
    # Initialize checker
    checker = DatasetAccessChecker()
    
    # Run access check
    results, accessible_count = checker.check_all_datasets(
        datasets_to_check, 
        test_samples=args.test_samples
    )
    
    # Print summary
    print_summary(results, len(datasets_to_check))
    
    # Save results if requested
    if args.output:
        output_data = {
            'check_timestamp': time.time(),
            'tested_samples_per_dataset': args.test_samples,
            'results': results
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    # Exit code based on success
    if accessible_count == len(datasets_to_check):
        print(f"\n🎉 All datasets are accessible!")
        return 0
    else:
        print(f"\n⚠️  {len(datasets_to_check) - accessible_count} datasets need attention")
        return 1


if __name__ == "__main__":
    exit(main())