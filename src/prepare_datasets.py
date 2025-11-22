"""Prepare all datasets for training with state tracking"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from core.dataset_processor import DatasetProcessor
from typing import Optional, List

from utils.config import TRAINING_STATE_FILE as STATE_FILE

class DatasetPreparationState:
    """Manage dataset preparation state"""
    
    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self.state = self._load_state()
    
    def _load_state(self) -> dict:
        """Load existing state or create new"""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "version": "1.0",
                "last_updated": None,
                "datasets": {}
            }
    
    def _save_state(self):
        """Save current state to file"""
        self.state["last_updated"] = datetime.now().isoformat()
        
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def is_prepared(self, dataset_name: str) -> bool:
        """Check if dataset is marked as prepared"""
        dataset_state = self.state["datasets"].get(dataset_name, {})
        return dataset_state.get("status") == "prepared"
    
    def mark_preparing(self, dataset_name: str):
        """Mark dataset as currently being prepared"""
        self.state["datasets"][dataset_name] = {
            "status": "preparing",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "train_samples": None,
            "test_samples": None,
            "size_gb": None,
            "error": None
        }
        self._save_state()
    
    def mark_prepared(self, dataset_name: str, train_num: int, test_num: int, size_gb: float):
        """Mark dataset as successfully prepared"""
        if dataset_name not in self.state["datasets"]:
            self.state["datasets"][dataset_name] = {}
        
        self.state["datasets"][dataset_name].update({
            "status": "prepared",
            "end_time": datetime.now().isoformat(),
            "train_samples": train_num,
            "test_samples": test_num,
            "total_samples": train_num + test_num,
            "size_gb": round(size_gb, 2),
            "error": None
        })
        self._save_state()
    
    def mark_failed(self, dataset_name: str, error_msg: str):
        """Mark dataset as failed"""
        if dataset_name not in self.state["datasets"]:
            self.state["datasets"][dataset_name] = {}
        
        self.state["datasets"][dataset_name].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": error_msg
        })
        self._save_state()
    
    def mark_skipped(self, dataset_name: str):
        """Mark dataset as skipped (already exists)"""
        if dataset_name not in self.state["datasets"]:
            self.state["datasets"][dataset_name] = {}
        
        self.state["datasets"][dataset_name].update({
            "status": "skipped",
            "skipped_time": datetime.now().isoformat()
        })
        self._save_state()
    
    def get_summary(self) -> dict:
        """Get summary statistics"""
        statuses = [ds.get("status") for ds in self.state["datasets"].values()]
        return {
            "total": len(self.state["datasets"]),
            "prepared": statuses.count("prepared"),
            "preparing": statuses.count("preparing"),
            "failed": statuses.count("failed"),
            "skipped": statuses.count("skipped")
        }
    
    def get_prepared_datasets(self) -> list:
        """Get list of successfully prepared datasets"""
        return [name for name, info in self.state["datasets"].items() 
                if info.get("status") == "prepared"]
    
    def get_failed_datasets(self) -> list:
        """Get list of failed datasets with errors"""
        return [(name, info.get("error")) 
                for name, info in self.state["datasets"].items() 
                if info.get("status") == "failed"]
    
    def reset_dataset(self, dataset_name: str):
        """Reset state for a specific dataset"""
        if dataset_name in self.state["datasets"]:
            del self.state["datasets"][dataset_name]
            self._save_state()
    
    def reset_all(self):
        """Reset all state (fresh start)"""
        self.state = {
            "version": "1.0",
            "last_updated": None,
            "datasets": {}
        }
        self._save_state()


def get_folder_size_gb(folder_path: str) -> float:
    """Calculate folder size in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size / (1024 ** 3)


def prepare_single_dataset(processor: DatasetProcessor, 
                          dataset_name: str, 
                          state: DatasetPreparationState,
                          force: bool = False) -> bool:
    """Prepare a single dataset with state tracking"""
    
    try:
        # Check state file first
        if not force and state.is_prepared(dataset_name):
            print(f"  ⊙ Already prepared (from state file), skipping")
            return True
        
        # Check if actually exists locally
        if processor.check_dataset_exists_locally(dataset_name):
            # Exists but not in state file - add to state
            train_num, test_num = processor._get_local_dataset_size(dataset_name)
            dataset_path = os.path.join(processor.local_dataset_base_path, dataset_name)
            size_gb = get_folder_size_gb(dataset_path)
            
            state.mark_prepared(dataset_name, train_num, test_num, size_gb)
            print(f"  ✓ Already exists locally, added to state file")
            return True
        
        # Mark as preparing
        state.mark_preparing(dataset_name)
        print(f"  → Copying dataset...")
        
        # Copy dataset
        start_time = time.time()
        processor.copy_dataset_to_local(dataset_name)
        duration = time.time() - start_time
        
        # Verify copy
        if not processor.check_dataset_exists_locally(dataset_name):
            raise Exception("Verification failed: file count mismatch after copy")
        
        # Get statistics
        train_num, test_num = processor._get_local_dataset_size(dataset_name)
        dataset_path = os.path.join(processor.local_dataset_base_path, dataset_name)
        size_gb = get_folder_size_gb(dataset_path)
        
        # Mark as prepared
        state.mark_prepared(dataset_name, train_num, test_num, size_gb)
        
        print(f"  ✓ Complete! Train: {train_num}, Test: {test_num}, "
              f"Size: {size_gb:.2f} GB, Time: {duration:.1f}s")
        return True
        
    except Exception as e:
        error_msg = str(e)
        state.mark_failed(dataset_name, error_msg)
        print(f"  ✗ ERROR: {error_msg}")
        return False


def print_state_summary(state: DatasetPreparationState):
    """Print current state summary"""
    summary = state.get_summary()
    
    print(f"\n{'='*70}")
    print("PREPARATION STATE SUMMARY")
    print(f"{'='*70}")
    print(f"Total datasets:     {summary['total']}")
    print(f"Prepared:          {summary['prepared']} ✓")
    print(f"Currently preparing: {summary['preparing']} ⟳")
    print(f"Failed:            {summary['failed']} ✗")
    print(f"Skipped:           {summary['skipped']} ⊙")
    
    if summary['failed'] > 0:
        print(f"\nFailed datasets:")
        for name, error in state.get_failed_datasets():
            print(f"  - {name}: {error}")
    
    print(f"{'='*70}\n")


def main(force_all: bool = False, reset: bool = False, specific_datasets: Optional[List[str]] = None):
    """Prepare all datasets locally with state tracking
    
    Args:
        force_all: Force re-preparation of all datasets
        reset: Reset state file before starting
        specific_datasets: List of specific datasets to prepare (None = all)
    """
    
    from utils.config import ALL_DATASETS as all_datasets
    
    # Determine which datasets to process
    datasets = specific_datasets if specific_datasets else all_datasets
    
    # Initialize state
    state = DatasetPreparationState()
    
    if reset:
        print("Resetting state file...")
        state.reset_all()
    
    # Print current state
    if not reset and state.state["datasets"]:
        print_state_summary(state)
    
    # Initialize processor
    processor = DatasetProcessor(datasets)
    
    print(f"{'='*70}")
    print("DATASET PREPARATION")
    print(f"{'='*70}")
    print(f"Datasets to process: {len(datasets)}")
    print(f"Target directory: {processor.local_dataset_base_path}")
    print(f"State file: {state.state_file}")
    if force_all:
        print("Mode: FORCE (re-preparing all datasets)")
    print(f"{'='*70}\n")
    
    # Process each dataset
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, dataset_name in enumerate(datasets, 1):
        print(f"[{i}/{len(datasets)}] Processing {dataset_name}...")
        
        result = prepare_single_dataset(processor, dataset_name, state, force=force_all)
        
        if result:
            if state.is_prepared(dataset_name):
                success_count += 1
            else:
                skip_count += 1
        else:
            fail_count += 1
    
    # Final summary
    print(f"\n{'='*70}")
    print("PREPARATION COMPLETE")
    print(f"{'='*70}")
    print(f"Successful:  {success_count} ✓")
    print(f"Skipped:     {skip_count} ⊙")
    print(f"Failed:      {fail_count} ✗")
    
    # Calculate total statistics for prepared datasets
    prepared_datasets = state.get_prepared_datasets()
    if prepared_datasets:
        total_samples = sum(
            state.state["datasets"][ds].get("total_samples", 0)
            for ds in prepared_datasets
        )
        total_size = sum(
            state.state["datasets"][ds].get("size_gb", 0)
            for ds in prepared_datasets
        )
        
        print(f"\nTotal samples: {total_samples:,}")
        print(f"Total size:    {total_size:.2f} GB")
    
    print(f"\nState file: {state.state_file}")
    print(f"Location:   {processor.local_dataset_base_path}")
    print(f"{'='*70}\n")
    
    # Print failed datasets if any
    if fail_count > 0:
        print("⚠️  Some datasets failed. Check errors above or run:")
        print(f"   cat {state.state_file}")
        print("\nTo retry failed datasets, run:")
        print("   python src/prepare_datasets.py --retry-failed")
        return False
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare datasets for training')
    parser.add_argument('--force', action='store_true',
                       help='Force re-preparation of all datasets')
    parser.add_argument('--reset', action='store_true',
                       help='Reset state file before starting')
    parser.add_argument('--retry-failed', action='store_true',
                       help='Retry only failed datasets')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to prepare (e.g., 010_RSNA 011_SIIM-ACR)')
    parser.add_argument('--show-state', action='store_true',
                       help='Show current state and exit')
    
    args = parser.parse_args()
    
    # Show state only
    if args.show_state:
        state = DatasetPreparationState()
        print_state_summary(state)
        
        print("\nDetailed state:")
        print(json.dumps(state.state, indent=2))
        sys.exit(0)
    
    # Retry failed datasets
    specific_datasets = None
    if args.retry_failed:
        state = DatasetPreparationState()
        failed_datasets = [name for name, _ in state.get_failed_datasets()]
        if not failed_datasets:
            print("No failed datasets to retry.")
            sys.exit(0)
        print(f"Retrying {len(failed_datasets)} failed datasets...")
        specific_datasets = failed_datasets
    elif args.datasets:
        specific_datasets = args.datasets
    
    # Run preparation
    success = main(
        force_all=args.force,
        reset=args.reset,
        specific_datasets=specific_datasets
    )
    
    sys.exit(0 if success else 1)