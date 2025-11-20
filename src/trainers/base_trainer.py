"""Base trainer class for all models"""
import os
import json
import torch
import wandb
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

class BaseTrainer(ABC):
    """Base class for model training"""
    
    def __init__(self, args, dataset_loader):
        self.args = args
        self.dataset_loader = dataset_loader
        
        # Setup device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # Setup save path
        from core import metrics
        self.model_save_path = os.path.join(
            metrics.SAVING_BASE_DIR,
            f"{args.model_name}_{args.model_size}_{args.dataset_name}"
        )
        os.makedirs(self.model_save_path, exist_ok=True)

        from core.training_logger import TrainingLogger
        self.logger = TrainingLogger()
        
        # Initialize model, optimizer, and scheduler
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Best model tracking
        self.best_acc = 0.0
        self.best_epoch = 0
        self.best_predictions = None
        self.best_ground_truth = None
    
    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        """Create and return the model"""
        pass
    
    @abstractmethod
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create and return the optimizer"""
        pass
    
    @abstractmethod
    def create_scheduler(self) -> Optional[object]:
        """Create and return the learning rate scheduler"""
        pass
    
    @abstractmethod
    def get_transforms(self) -> Dict:
        """Return train and test transforms"""
        pass
    
    def initialize_wandb(self):
        """Initialize Weights & Biases logging"""
        config = {
            "model_name": self.args.model_name,
            "model_size": self.args.model_size,
            "dataset_name": self.args.dataset_name,
            "batch_size": self.args.batch_size,
            "learning_rate": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
            "epochs": self.args.epochs,
            "device": str(self.device)
        }
        
        if self.args.wandb_name:
            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                name=self.args.wandb_name,
                config=config
            )
        else:
            wandb.init(
                project=self.args.wandb_project,
                entity=self.args.wandb_entity,
                config=config
            )
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs} [Train]')
        for batch_idx, (images, labels) in enumerate(train_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            train_correct = (np.array(train_predictions) == np.array(train_targets)).sum()
            train_total = len(train_targets)
            train_bar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        return train_loss / len(train_loader), train_predictions, train_targets
    
    def evaluate(self, test_loader, epoch):
        """Evaluate on test set"""
        self.model.eval()
        test_loss = 0.0
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            test_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{self.args.epochs} [Test]')
            for batch_idx, (images, labels) in enumerate(test_bar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                
                test_predictions.extend(predicted.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
                
                # Update progress bar
                test_correct = (np.array(test_predictions) == np.array(test_targets)).sum()
                test_total = len(test_targets)
                test_bar.set_postfix({
                    'loss': f'{test_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*test_correct/test_total:.2f}%'
                })
        
        return test_loss / len(test_loader), test_predictions, test_targets
    
    def calculate_and_log_metrics(self, train_predictions, train_targets, 
                                  test_predictions, test_targets, epoch, 
                                  train_loss, test_loss):
        """Calculate metrics and log to wandb"""
        from core import metrics
        
        task_type = self.dataset_loader.get_task_type()
        
        if task_type == 'binary':
            train_acc, train_f1, _, _ = metrics.calculate_metrics(
                train_predictions, train_targets, self.dataset_loader, self.args
            )
            test_acc, test_f1, _, _ = metrics.calculate_metrics(
                test_predictions, test_targets, self.dataset_loader, self.args
            )
            
            train_metrics = {'accuracy': float(train_acc), 'f1_score': float(train_f1)}
            test_metrics = {'accuracy': float(test_acc), 'f1_score': float(test_f1)}
            
            wandb_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': float(train_acc),
                'train_f1': float(train_f1),
                'test_loss': test_loss,
                'test_acc': float(test_acc),
                'test_f1': float(test_f1),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
        else:  # multiclass or multilabel
            train_acc, train_macro_f1, train_micro_f1, _, _ = metrics.calculate_metrics(
                train_predictions, train_targets, self.dataset_loader, self.args
            )
            test_acc, test_macro_f1, test_micro_f1, _, _ = metrics.calculate_metrics(
                test_predictions, test_targets, self.dataset_loader, self.args
            )
            
            train_metrics = {
                'accuracy': float(train_acc),
                'macro_f1': float(train_macro_f1),
                'micro_f1': float(train_micro_f1)
            }
            test_metrics = {
                'accuracy': float(test_acc),
                'macro_f1': float(test_macro_f1),
                'micro_f1': float(test_micro_f1)
            }
            
            wandb_metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': float(train_acc),
                'train_macro_f1': float(train_macro_f1),
                'train_micro_f1': float(train_micro_f1),
                'test_loss': test_loss,
                'test_acc': float(test_acc),
                'test_macro_f1': float(test_macro_f1),
                'test_micro_f1': float(test_micro_f1),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
        
        wandb.log(wandb_metrics)
        return train_metrics, test_metrics
    
    def save_best_model(self, epoch, train_metrics, test_metrics, 
                       test_predictions, test_targets):
        """Save the best model checkpoint and results"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_acc,
            'args': vars(self.args),
        }, os.path.join(self.model_save_path, 'best_model.pth'))
        
        # Save results JSON
        label_set = self.dataset_loader.classes
        best_results = {
            'epoch': epoch,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'class_names': label_set,
            'class_to_idx': self.dataset_loader.get_class_to_idx(),
            'predictions': []
        }
        
        for i, (pred, gt) in enumerate(zip(test_predictions, test_targets)):
            best_results['predictions'].append({
                'sample_id': i,
                'ground_truth': int(gt),
                'ground_truth_class': label_set[gt],
                'prediction': int(pred),
                'prediction_class': label_set[pred],
                'correct': bool(pred == gt)
            })
        
        with open(os.path.join(self.model_save_path, 'best_result.json'), 'w') as f:
            json.dump(best_results, f, indent=4)
    
    def train(self):
        """Main training loop"""
        # Check if already trained
        from core.training_logger import check_already_trained
        if check_already_trained(self.args.model_name, self.args.model_size, self.args.dataset_name):
            print(f"✅ Model {self.args.model_name}_{self.args.model_size} already trained on {self.args.dataset_name}")
            return
        
        # Initialize components
        self.model = self.create_model()
        self.model.to(self.device)
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Get data loaders
        transforms_dict = self.get_transforms()
        train_loader = self.dataset_loader.get_train_loader(transforms_dict['train'])
        test_loader = self.dataset_loader.get_test_loader(transforms_dict['test'])
        
        dataset_info = {
                'num_classes': len(self.dataset_loader.classes),
                'train_samples': len(train_loader.dataset),
                'test_samples': len(test_loader.dataset),
                'label_set': self.dataset_loader.classes,
                'save_path': self.model_save_path
            }
        self.session_id = self.logger.create_session(self.args, dataset_info)

        print(f"🚀 Starting training for {self.args.model_name}_{self.args.model_size} on {self.args.dataset_name}")
        print(f"📊 Classes: {len(self.dataset_loader.classes)}")
        print(f"📁 Train samples: {len(train_loader.dataset)}")
        print(f"📁 Test samples: {len(test_loader.dataset)}")
        print(f"💾 Save path: {self.model_save_path}")
        print(f"📝 Session ID: {self.session_id}")

        try:
            # Initialize wandb
            self.initialize_wandb()
            
            # Training loop
            for epoch in range(self.args.epochs):
                # Train
                train_loss, train_predictions, train_targets = self.train_epoch(train_loader, epoch)
                
                # Evaluate
                test_loss, test_predictions, test_targets = self.evaluate(test_loader, epoch)
                
                # No scheduler step - as in the original papers
                
                # Calculate and log metrics
                train_metrics, test_metrics = self.calculate_and_log_metrics(
                    train_predictions, train_targets,
                    test_predictions, test_targets,
                    epoch, train_loss, test_loss
                )
                
                # Save best model
                if test_metrics['accuracy'] > self.best_acc:
                    self.best_acc = test_metrics['accuracy']
                    self.best_epoch = epoch + 1
                    self.best_predictions = test_predictions.copy()
                    self.best_ground_truth = test_targets.copy()
                    self.save_best_model(
                        self.best_epoch, train_metrics, test_metrics,
                        test_predictions, test_targets
                    )
                
            final_results = {
                'best_accuracy': self.best_acc,
                'best_epoch': self.best_epoch,
                'final_train_metrics': train_metrics,
                'final_test_metrics': test_metrics
            }
            self.logger.complete_session(self.session_id, final_results)
        
            print(f"\nTraining completed! Best accuracy: {self.best_acc:.4f} at epoch {self.best_epoch}")
        
        except Exception as e:        
            error_msg = str(e)
            self.logger.fail_session(self.session_id, error_msg)
            print(f"❌ Training failed: {error_msg}")
            raise
        
        finally:
            wandb.finish()

            from core.training_logger import print_training_summary
            print_training_summary()
