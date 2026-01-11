import os
import json
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import logging
from typing import Dict, List, Optional, Any

from dataset.dataset import LaMPDataset, collate_fn
from metrics import compute_metric

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelSaver:
    """Handles saving models and metrics with improved error handling"""
    
    def __init__(self, accelerator, base_dir="saved"):
        self.accelerator = accelerator
        self.base_dir = Path(base_dir)
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.metrics_dir = self.base_dir / "metrics"
        
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        try:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            raise
    
    def get_timestamp(self):
        """Get current timestamp for unique filenames"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_model(self, model, exp_name="exp", task_name=None, epoch=None, step=None):
        """Save model checkpoint with error handling"""
        try:
            timestamp = self.get_timestamp()
            
            if task_name and epoch and step:
                filename = f"{exp_name}_{task_name}_epoch_{epoch}_step_{step}_{timestamp}.pt"
            elif task_name and epoch:
                filename = f"{exp_name}_{task_name}_epoch_{epoch}_{timestamp}.pt"
            else:
                filename = f"{exp_name}_{timestamp}.pt"
            
            save_path = self.checkpoints_dir / filename
            
            model_state = self.accelerator.unwrap_model(model).state_dict()
            torch.save(model_state, save_path)
            
            self.accelerator.print(f"Model saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def save_metrics(self, metrics: Dict[str, Any], task_name: Optional[str] = None, exp_name: Optional[str] = None):
        """Save metrics to JSON file with improved efficiency"""
        try:
            if task_name and exp_name:
                filename = f"metrics_{task_name}_{exp_name}.json"
            elif task_name:
                filename = f"metrics_{task_name}.json"
            elif exp_name:
                filename = f"metrics_{exp_name}.json"
            else:
                filename = f"metrics.json"
            
            save_path = self.metrics_dir / filename

            # Load existing data or create new
            entries = []
            if save_path.exists():
                try:
                    with open(save_path, 'r', encoding='utf-8') as f:
                        entries = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Could not load existing metrics file, creating new: {e}")
            
            # Add new entry
            entries.append({
                "timestamp": self.get_timestamp(),
                "metrics": metrics
            })
            
            # Save updated data
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(entries, f, indent=4, ensure_ascii=False)
            
            self.accelerator.print(f"Metrics saved to {save_path} (total entries: {len(entries)})")
            return save_path
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            raise


class TaskSequentialTrainer:
    """Trainer for sequential task learning with fixes applied"""
    
    def __init__(self, accelerator, model, optimizer, llm_tokenizer, beh_tokenizer, input_tokenizer, max_len_llm, max_len_beh, llm_type, exp_name):
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = None
        self.llm_tokenizer = llm_tokenizer
        self.beh_tokenizer = beh_tokenizer
        self.input_tokenizer = input_tokenizer
        self.max_len_llm = max_len_llm
        self.max_len_beh = max_len_beh
        self.llm_type = llm_type
        self.exp_name = exp_name
        self.saver = ModelSaver(accelerator)
        
        # Generation parameters - consistent across evaluation
        self.generation_params = {
            "max_new_tokens": 3,
            "do_sample": False,
            "repetition_penalty": 1.2,
            "num_return_sequences": 1,
            "eos_token_id": self.llm_tokenizer.eos_token_id,
            "pad_token_id": self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
        }
    
    def train(self, tasks: Dict[str, Dict], num_epochs: int, batch_size: int, warmup_ratio: float):
        """Train model sequentially on each task"""
        for task_name, task_info in tasks.items():
            tasks_metrics = {}
            self.accelerator.print(f"Starting training on task: {task_name}")

            # Prepare data loaders
            train_loader, val_loader = self._prepare_data_loaders(task_info, batch_size)
            
            # Calculate scheduler parameters
            steps_per_task = num_epochs * len(train_loader)
            warmup_steps = int(warmup_ratio * steps_per_task)
            
            # Create scheduler BEFORE preparing
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=steps_per_task
            )
            
            # Prepare all components together
            self.model, self.optimizer, self.scheduler, train_loader, val_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.scheduler, train_loader, val_loader
            )
            
            # Log trainable parameters (only once)
            if hasattr(self, '_logged_params') == False:
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
                self.accelerator.print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
                self._logged_params = True

            # Train on current task
            self._train_task(train_loader, val_loader, task_name, task_info, num_epochs)

            # Evaluate after each task
            self.model.eval()
            metrics = self.evaluate(val_loader, task_info["metric"])
            tasks_metrics[task_name] = metrics
            self.accelerator.print(f"Metrics for {task_name} ({task_info['metric']}): {metrics}")

            # Save after each task
            if self.accelerator.is_local_main_process:
                self.saver.save_model(self.model, self.exp_name, task_name, num_epochs)
            if self.accelerator.is_local_main_process and metrics is not None:
                self.saver.save_metrics(tasks_metrics, task_name=task_name, exp_name=self.exp_name)
                
        self.accelerator.print("Sequential training completed.")
        return self.model
    
    def _prepare_data_loaders(self, task_info: Dict, batch_size: int):
        """Prepare train and validation data loaders"""
        train_dataset = LaMPDataset(
            json_path=task_info["train_json_path"],
            llm_tokenizer=self.llm_tokenizer,
            beh_tokenizer=self.beh_tokenizer,
            input_tokenizer=self.input_tokenizer,
            max_len_llm=self.max_len_llm,
            max_len_beh=self.max_len_beh,
            llm_type=self.llm_type
        )
        
        val_dataset = LaMPDataset(
            json_path=task_info["val_json_path"],
            llm_tokenizer=self.llm_tokenizer,
            beh_tokenizer=self.beh_tokenizer,
            input_tokenizer=self.input_tokenizer,
            max_len_llm=self.max_len_llm,
            max_len_beh=self.max_len_beh,
            llm_type=self.llm_type
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=2,  # Add parallel loading
            pin_memory=True  # Faster GPU transfer
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _train_task(self, train_loader, val_loader, task_name: str, task_info: Dict, num_epochs: int):
        """Train model on a single task"""
        global_step = 0
        save_every_steps = 20_000
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{num_epochs}", 
                disable=not self.accelerator.is_local_main_process
            )
            
            self.model.train()
            for batch in progress_bar:
                loss = self._process_batch(batch)
                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss/num_batches:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })

                # # Save checkpoint
                # if self.accelerator.is_local_main_process and global_step % save_every_steps == 0:
                #     self.saver.save_model(
                #         self.model,
                #         self.exp_name,
                #         task_name,
                #         epoch+1,
                #         global_step
                #     )

            # Calculate and log average loss
            avg_loss = total_loss / num_batches
            self.accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

            # # Save after each epoch
            # if self.accelerator.is_local_main_process:
            #     self.saver.save_model(self.model, self.exp_name, task_name, epoch, global_step)
                
    def _process_batch(self, batch):
        """Process a single batch and update model with proper gradient handling"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "llm_input_ids": batch["llm_input_ids"], 
            "llm_attention_mask": batch["llm_attention_mask"],
            "beh_input_ids": batch["beh_input_ids"],
            "beh_attention_mask": batch["beh_attention_mask"],
            "labels": batch["labels"]
        }
        
        # Use autocast for mixed precision
        with self.accelerator.autocast():
            outputs = self.model(**inputs)
            loss = outputs.loss

        # Backward pass with gradient clipping
        self.accelerator.backward(loss)
        
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step() 
        self.optimizer.zero_grad()
        
        return loss
        
    def evaluate(self, val_loader, metric_name: str):
        """Evaluate model on validation data"""
        all_preds, all_labels = [], []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", disable=not self.accelerator.is_local_main_process):
                generated_ids = self.model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    llm_input_ids=batch["llm_input_ids"],
                    llm_attention_mask=batch["llm_attention_mask"],
                    beh_input_ids=batch["beh_input_ids"],
                    beh_attention_mask=batch["beh_attention_mask"],
                    **self.generation_params
                )
                
                # Synchronize across processes
                generated_ids = self.accelerator.gather(
                    self.accelerator.pad_across_processes(generated_ids, dim=1)
                )
                labels = self.accelerator.gather(
                    self.accelerator.pad_across_processes(batch["labels"], dim=1)
                )
                
                # Process on main process only
                if self.accelerator.is_local_main_process:
                    preds_text = self.llm_tokenizer.batch_decode(
                        generated_ids.cpu(), 
                        skip_special_tokens=True
                    )
                    
                    # Handle label padding
                    labels[labels == -100] = self.llm_tokenizer.pad_token_id
                    labels_text = self.llm_tokenizer.batch_decode(
                        labels.cpu(),
                        skip_special_tokens=True
                    )

                    all_preds.extend(preds_text)
                    all_labels.extend(labels_text)
        
        if self.accelerator.is_local_main_process:
            return compute_metric(metric_name, all_preds, all_labels, self.llm_tokenizer)
        return None


class JointTaskTrainer:
    """Trainer for joint learning with fixes applied"""
    
    def __init__(self, accelerator, model, optimizer, llm_tokenizer, beh_tokenizer, input_tokenizer, max_len_llm, max_len_beh, llm_type, exp_name):
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = None
        self.llm_tokenizer = llm_tokenizer
        self.beh_tokenizer = beh_tokenizer
        self.input_tokenizer = input_tokenizer
        self.max_len_llm = max_len_llm
        self.max_len_beh = max_len_beh
        self.llm_type = llm_type
        self.exp_name = exp_name
        self.saver = ModelSaver(accelerator)
    
    def train(self, json_path: str, num_epochs: int, batch_size: int, warmup_ratio: float):
        """Train model on combined dataset"""
        train_loader = self._prepare_data_loader(json_path, batch_size)

        total_training_steps = len(train_loader) * num_epochs
        warmup_steps = int(warmup_ratio * total_training_steps)
        
        # Create scheduler before preparing
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_training_steps
        )
        
        # Prepare all components
        self.model, self.optimizer, self.scheduler, train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, train_loader
        )
        
        # Log parameter info
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        self.accelerator.print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        self.accelerator.print(f"Frozen params: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{num_epochs}", 
                disable=not self.accelerator.is_local_main_process
            )
            
            self.model.train()
            for batch in progress_bar:
                loss = self._process_batch(batch)
                total_loss += loss.item()
                num_batches += 1

                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{total_loss/num_batches:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
            
            avg_loss = total_loss / num_batches
            self.accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
            
            if self.accelerator.is_local_main_process:
                self.saver.save_model(self.model, self.exp_name, epoch=epoch+1)
                
        self.accelerator.print("Joint training completed.")
        return self.model
    
    def _prepare_data_loader(self, json_path: str, batch_size: int):
        """Prepare combined data loader"""
        dataset = LaMPDataset(
            json_path=json_path,
            llm_tokenizer=self.llm_tokenizer,
            beh_tokenizer=self.beh_tokenizer,
            input_tokenizer=self.input_tokenizer,
            max_len_llm=self.max_len_llm,
            max_len_beh=self.max_len_beh,
            llm_type=self.llm_type
        )
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
    
    def _process_batch(self, batch):
        """Process a single batch and update model"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "llm_input_ids": batch["llm_input_ids"],
            "llm_attention_mask": batch["llm_attention_mask"],
            "beh_input_ids": batch["beh_input_ids"],
            "beh_attention_mask": batch["beh_attention_mask"],
            "labels": batch["labels"]
        }

        with self.accelerator.autocast():
            outputs = self.model(**inputs)
        loss = outputs.loss
        
        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss


class ModelEvaluator:
    """Evaluator for model performance with consistent parameters"""
    
    def __init__(self, accelerator, model, llm_tokenizer, beh_tokenizer, input_tokenizer, max_len_llm, max_len_beh, llm_type, exp_name):
        self.accelerator = accelerator
        self.model = model
        self.llm_tokenizer = llm_tokenizer
        self.beh_tokenizer = beh_tokenizer
        self.input_tokenizer = input_tokenizer
        self.max_len_llm = max_len_llm
        self.max_len_beh = max_len_beh
        self.llm_type = llm_type
        self.exp_name = exp_name
        self.saver = ModelSaver(accelerator)
        
        # Consistent generation parameters
        self.generation_params = {
            "max_new_tokens": 3,
            "do_sample": False,
            "repetition_penalty": 1.2,
            "num_return_sequences": 1,
            "eos_token_id": self.llm_tokenizer.eos_token_id,
            "pad_token_id": self.llm_tokenizer.pad_token_id or self.llm_tokenizer.eos_token_id,
        }
    
    def evaluate(self, tasks: Dict[str, Dict], batch_size: int = 64):
        """Evaluate model on all specified tasks"""
        tasks_metrics = {}
        
        for task_name, task_info in tasks.items():
            self.accelerator.print(f"Evaluating on task: {task_name}")
            
            val_loader = self._prepare_data_loader(task_info["val_json_path"], batch_size)
            val_loader = self.accelerator.prepare(val_loader)

            metrics = self._evaluate_task(val_loader, task_info["metric"])
            tasks_metrics[task_name] = metrics

            self.accelerator.print(f"Metrics for {task_name} ({task_info['metric']}): {metrics}")

        # Save all metrics together
        if self.accelerator.is_local_main_process and tasks_metrics:
            self.saver.save_metrics(tasks_metrics, exp_name=self.exp_name)
        return tasks_metrics
    
    def _prepare_data_loader(self, json_path: str, batch_size: int):
        """Prepare data loader for evaluation"""
        dataset = LaMPDataset(
            json_path=json_path,
            llm_tokenizer=self.llm_tokenizer,
            beh_tokenizer=self.beh_tokenizer,
            input_tokenizer=self.input_tokenizer,
            max_len_llm=self.max_len_llm,
            max_len_beh=self.max_len_beh,
            llm_type=self.llm_type
        )
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
    def _evaluate_task(self, val_loader, metric_name: str):
        """Evaluate model on a single task"""
        all_preds, all_labels = [], []

        self.model.eval()
        with self.accelerator.autocast():
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating", 
                                 disable=not self.accelerator.is_local_main_process):
                    
                    generated_ids = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        llm_input_ids=batch["llm_input_ids"],
                        llm_attention_mask=batch["llm_attention_mask"],
                        beh_input_ids=batch["beh_input_ids"],
                        beh_attention_mask=batch["beh_attention_mask"],
                        **self.generation_params  # Use consistent parameters
                    )
                    
                    # Synchronize across processes
                    generated_ids = self.accelerator.gather(
                        self.accelerator.pad_across_processes(generated_ids, dim=1)
                    )
                    labels = self.accelerator.gather(
                        self.accelerator.pad_across_processes(batch["labels"], dim=1)
                    )
                    
                    # Process on main process only
                    if self.accelerator.is_local_main_process:
                        preds_text = self.llm_tokenizer.batch_decode(
                            generated_ids.cpu(), 
                            skip_special_tokens=True
                        )
                        
                        # Handle label padding
                        labels[labels == -100] = self.llm_tokenizer.pad_token_id
                        labels_text = self.llm_tokenizer.batch_decode(
                            labels.cpu(),
                            skip_special_tokens=True
                        )
                        print(f"preds_text = {preds_text}")
                        print(f"labels_text = {labels_text}")
                        all_preds.extend(preds_text)
                        all_labels.extend(labels_text)
            
        if self.accelerator.is_local_main_process:
            return compute_metric(metric_name, all_preds, all_labels, self.llm_tokenizer)
        return None