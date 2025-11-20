"""
Training logger for tracking model training sessions and results
"""

import json
import os
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch

class TrainingLogger:
    """JSON-based training session logger"""
    
    def __init__(self, log_dir: str = "./logs/model_training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.json_log = self.log_dir / "training_history.json"
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Initialize JSON log file if it doesn't exist"""
        if not self.json_log.exists():
            initial_data = {
                "version": "1.0",
                "created_at": datetime.datetime.now().isoformat(),
                "training_sessions": []
            }
            self._write_json(initial_data)
    
    def _read_json(self) -> Dict[str, Any]:
        """Read JSON log file"""
        try:
            with open(self.json_log, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted, create new one
            self._initialize_log_file()
            return self._read_json()
    
    def _write_json(self, data: Dict[str, Any]):
        """Write data to JSON log file"""
        with open(self.json_log, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def create_session(self, args, dataset_info: Dict[str, Any]) -> int:
        """Create a new training session and return session ID"""
        data = self._read_json()
        
        session_id = len(data["training_sessions"]) + 1
        
        session = {
            "session_id": session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "started",
            "model": {
                "name": getattr(args, 'model_name', 'unknown'),
                "size": getattr(args, 'model_size', 'unknown')
            },
            "dataset": {
                "name": getattr(args, 'dataset_name', 'unknown'),
                "num_classes": dataset_info.get('num_classes', 0),
                "train_samples": dataset_info.get('train_samples', 0),
                "test_samples": dataset_info.get('test_samples', 0),
                "label_set": dataset_info.get('label_set', [])
            },
            "training_config": {
                "epochs": getattr(args, 'epochs', 0),
                "batch_size": getattr(args, 'batch_size', 0),
                "learning_rate": getattr(args, 'learning_rate', 0),
                "weight_decay": getattr(args, 'weight_decay', 0),
                "device": getattr(args, 'device', 'cpu'),
                "num_workers": getattr(args, 'num_workers', 4)
            },
            "save_path": dataset_info.get('save_path', ''),
            "results": {},
            "error_message": None
        }
        
        data["training_sessions"].append(session)
        self._write_json(data)
        
        print(f"📝 Training session created: ID {session_id}")
        return session_id
    
    def update_session(self, session_id: int, updates: Dict[str, Any]):
        """Update an existing training session"""
        data = self._read_json()
        
        for session in data["training_sessions"]:
            if session["session_id"] == session_id:
                session.update(updates)
                # Always update timestamp when modifying
                session["timestamp"] = datetime.datetime.now().isoformat()
                break
        
        self._write_json(data)
    
    def complete_session(self, session_id: int, results: Dict[str, Any]):
        """Mark a session as completed with results"""
        self.update_session(session_id, {
            "status": "completed",
            "results": results
        })
        print(f"✅ Training session {session_id} completed")
    
    def fail_session(self, session_id: int, error_msg: str):
        """Mark a session as failed"""
        self.update_session(session_id, {
            "status": "failed",
            "error_message": error_msg
        })
        print(f"❌ Training session {session_id} failed: {error_msg}")
    
    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific session by ID"""
        data = self._read_json()
        for session in data["training_sessions"]:
            if session["session_id"] == session_id:
                return session
        return None
    
    def find_sessions(self, model_name: Optional[str] = None, model_size: Optional[str] = None, 
                     dataset_name: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find sessions matching criteria"""
        data = self._read_json()
        sessions = data["training_sessions"]
        
        if model_name:
            sessions = [s for s in sessions if s["model"]["name"] == model_name]
        if model_size:
            sessions = [s for s in sessions if s["model"]["size"] == model_size]
        if dataset_name:
            sessions = [s for s in sessions if s["dataset"]["name"] == dataset_name]
        if status:
            sessions = [s for s in sessions if s["status"] == status]
        
        return sessions
    
    def is_already_trained(self, model_name: str, model_size: str, dataset_name: str) -> bool:
        """Check if a model has been successfully trained on a dataset"""
        completed_sessions = self.find_sessions(
            model_name=model_name,
            model_size=model_size,
            dataset_name=dataset_name,
            status="completed"
        )
        return len(completed_sessions) > 0
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all training sessions"""
        data = self._read_json()
        sessions = data["training_sessions"]
        
        total_sessions = len(sessions)
        completed = len([s for s in sessions if s["status"] == "completed"])
        failed = len([s for s in sessions if s["status"] == "failed"])
        running = len([s for s in sessions if s["status"] == "started"])
        
        return {
            "total_sessions": total_sessions,
            "completed": completed,
            "failed": failed,
            "running": running,
            "success_rate": completed / total_sessions if total_sessions > 0 else 0
        }


# Utility functions for easy access
def get_logger():
    """Get default training logger"""
    return TrainingLogger()

def check_already_trained(model_name: str, model_size: str, dataset_name: str) -> bool:
    """Convenience function to check if model is already trained"""
    logger = get_logger()
    return logger.is_already_trained(model_name, model_size, dataset_name)

def print_training_summary():
    """Print summary of all training sessions"""
    logger = get_logger()
    summary = logger.get_training_summary()
    
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Total sessions: {summary['total_sessions']}")
    print(f"Completed: {summary['completed']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Running: {summary['running']} ⏳")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"{'='*50}")