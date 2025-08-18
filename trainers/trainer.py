import os
import json
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset.dataset import LaMPDataset, collate_fn
from metrics import compute_metric


class ModelSaver:
    """Handles saving models and metrics"""
    
    def __init__(self, accelerator, base_dir="saved"):
        self.accelerator = accelerator
        self.base_dir = Path(base_dir)
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.metrics_dir = self.base_dir / "metrics"
        
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
    
    def get_timestamp(self):
        """Get current timestamp for unique filenames"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_model(self, model, task_name=None, epoch=None):
        """Save model checkpoint"""
        timestamp = self.get_timestamp()
        
        if task_name and epoch:
            filename = f"model_{task_name}_epoch{epoch}_{timestamp}.pt"
        else:
            filename = f"model_joint_{timestamp}.pt"
        
        save_path = self.checkpoints_dir / filename
        torch.save(self.accelerator.unwrap_model(model).state_dict(), save_path)
        self.accelerator.print(f"Model saved to {save_path}")
    
    def save_metrics(self, metrics, task_name=None):
        """Save metrics to JSON file"""
        timestamp = self.get_timestamp()
        
        if task_name:
            filename = f"metrics_{task_name}_{timestamp}.json"
        else:
            filename = f"metrics_joint_{timestamp}.json"
        
        save_path = self.metrics_dir / filename
        
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.accelerator.print(f"Metrics saved to {save_path}")
        

class TaskSequentialTrainer:
    """Trainer for sequential task learning (one task at a time)"""
    
    def __init__(self, accelerator, model, optimizer, llm_tokenizer, beh_tokenizer):
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = None
        self.llm_tokenizer = llm_tokenizer
        self.beh_tokenizer = beh_tokenizer
        self.saver = ModelSaver(accelerator)
    
    def train(self, tasks, num_epochs, batch_size, warmup_ratio):
        """Train model sequentially on each task"""
        for task_name, task_info in tasks.items():
            tasks_metrics = {}
            # if self.accelerator.is_local_main_process:
            self.accelerator.print(f"Starting training on task: {task_name}")

            # Prepare data loaders
            train_loader, val_loader = self._prepare_data_loaders(
                task_info["json_path"], 
                batch_size
            )
            steps_per_task = num_epochs * len(train_loader)
            warmup_steps = int(warmup_ratio * steps_per_task)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                            num_warmup_steps=warmup_steps, 
                                                            num_training_steps=steps_per_task
                                                            )
            # self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(self.model, self.optimizer, train_loader, val_loader)
            self.model, self.optimizer, self.scheduler, train_loader, val_loader = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, train_loader, val_loader)

            # train_loader, val_loader = self.accelerator.prepare(train_loader, val_loader)
            
            # Train on current task
            self.model.train()
            self._train_task(train_loader, num_epochs)
            
            # Evaluate after each task
            self.model.eval()
            metrics = self.evaluate(val_loader, task_info["metric"])
            tasks_metrics[task_name] = metrics
            self.accelerator.print(f"Metrics for {task_name} ({task_info['metric']}): {metrics}")

            # Save after each task
            if self.accelerator.is_local_main_process:
                self.saver.save_model(self.model, task_name, num_epochs)
            if self.accelerator.is_local_main_process and metrics is not None:
                self.saver.save_metrics(tasks_metrics)
        # if self.accelerator.is_local_main_process:
        self.accelerator.print("Sequential training completed.")
        return self.model
    
    def _prepare_data_loaders(self, train_json_path, batch_size):
        """Prepare train and validation data loaders"""
        train_dataset = LaMPDataset(
            json_path=train_json_path,
            llm_tokenizer=self.llm_tokenizer,
            beh_tokenizer=self.beh_tokenizer
        )
        
        val_dataset = LaMPDataset(
            json_path=train_json_path.replace("train", "dev"),
            llm_tokenizer=self.llm_tokenizer,
            beh_tokenizer=self.beh_tokenizer
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader
    
    def _train_task(self, train_loader, num_epochs):
        """Train model on a single task"""
        
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not self.accelerator.is_local_main_process)
            
            for batch in progress_bar:
                loss = self._process_batch(batch)
                total_loss += loss.item()
                
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({"loss": loss.item()})

            # Синхронизация loss между процессами
            avg_loss = torch.tensor(total_loss / len(train_loader)).to(self.accelerator.device)
            avg_loss = self.accelerator.gather(avg_loss).mean().item()
            
            if self.accelerator.is_local_main_process:
                self.accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    def _process_batch(self, batch):
        """Process a single batch and update model"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "beh_input_ids": batch["beh_input_ids"],
            "beh_attention_mask": batch["beh_attention_mask"],
            "labels": batch["labels"]
        }
        
        outputs = self.model(**inputs)
        loss = outputs.loss
        
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss
    
    def evaluate(self, val_loader, metric_name):
        """Evaluate model on validation data"""
        all_preds, all_labels = [], []

        with self.accelerator.autocast():
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating", disable=not self.accelerator.is_local_main_process):
                    generated_ids = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        beh_input_ids=batch["beh_input_ids"],
                        beh_attention_mask=batch["beh_attention_mask"],
                        max_length=150,
                        temperature=0.7,
                        num_beams=5,
                        early_stopping=True
                    )
                    
                    # Синхронизация
                    generated_ids = self.accelerator.gather(
                        self.accelerator.pad_across_processes(generated_ids, dim=1)
                    )
                    labels = self.accelerator.gather(
                        self.accelerator.pad_across_processes(batch["labels"], dim=1)
                    )
                    
                    # Только на главном процессе
                    if self.accelerator.is_local_main_process:
                        preds_text = self.llm_tokenizer.batch_decode(
                            generated_ids.cpu(), 
                            skip_special_tokens=True
                        )
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
    """Trainer for joint learning on all tasks simultaneously"""
    
    def __init__(self, accelerator, model, optimizer, llm_tokenizer, beh_tokenizer):
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.scheduler = None
        self.llm_tokenizer = llm_tokenizer
        self.beh_tokenizer = beh_tokenizer
        self.saver = ModelSaver(accelerator)
    
    def train(self, json_path, num_epochs, batch_size, warmup_ratio):
        """Train model on combined dataset"""
        train_loader = self._prepare_data_loader(json_path, batch_size)

        total_training_steps = len(train_loader)
        warmup_steps = int(warmup_ratio * total_training_steps)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                        num_warmup_steps=warmup_steps, 
                                                        num_training_steps=total_training_steps
                                                       )
        self.model, self.optimizer, self.scheduler, train_loader = self.accelerator.prepare(self.model, self.optimizer, self.scheduler, train_loader)
        
        self.model.train()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"Trainable params: {trainable_params} ({trainable_params/total_params*100:.2f}%)")
        print(f"Frozen params: {frozen_params} ({frozen_params/total_params*100:.2f}%)")
        
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not self.accelerator.is_local_main_process)
            
            for batch in progress_bar:
                loss = self._process_batch(batch)
                total_loss += loss.item()

                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({"loss": loss.item()})
            
            # Синхронизация loss
            avg_loss = torch.tensor(total_loss / len(train_loader)).to(self.accelerator.device)
            avg_loss = self.accelerator.gather(avg_loss).mean().item()
            
            if self.accelerator.is_local_main_process:
                self.accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
                self.saver.save_model(self.model, epoch=epoch+1)
        self.accelerator.print("Joint training completed.")
        return self.model
    
    def _prepare_data_loader(self, json_path, batch_size):
        """Prepare combined data loader"""
        dataset = LaMPDataset(
            json_path=json_path,
            llm_tokenizer=self.llm_tokenizer,
            beh_tokenizer=self.beh_tokenizer
        )
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn
        )
    
    def _process_batch(self, batch):
        """Process a single batch and update model"""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "beh_input_ids": batch["beh_input_ids"],
            "beh_attention_mask": batch["beh_attention_mask"],
            "labels": batch["labels"]
        }
        
        outputs = self.model(**inputs)
        loss = outputs.loss
        
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss


class ModelEvaluator:
    """Evaluator for model performance on multiple tasks"""
    
    def __init__(self, accelerator, model, llm_tokenizer, beh_tokenizer):
        self.accelerator = accelerator
        self.model = model
        self.llm_tokenizer = llm_tokenizer
        self.beh_tokenizer = beh_tokenizer
        self.saver = ModelSaver(accelerator)
    
    def evaluate(self, tasks, batch_size=64):
        """Evaluate model on all specified tasks"""
        tasks_metrics = {}
        
        for task_name, task_info in tasks.items():
            self.accelerator.print(f"Evaluating on task: {task_name}")
            
            val_loader = self._prepare_data_loader(
                task_info["json_path"].replace("train", "dev"),
                batch_size
            )

            val_loader = self.accelerator.prepare(val_loader)

            metrics = self._evaluate_task(val_loader, task_info["metric"])
            tasks_metrics[task_name] = metrics

            self.accelerator.print(f"Metrics for {task_name} ({task_info['metric']}): {metrics}")

        # Save all metrics together
        if self.accelerator.is_local_main_process and metrics is not None:
            self.saver.save_metrics(tasks_metrics)
        return tasks_metrics
    
    def _prepare_data_loader(self, json_path, batch_size):
        """Prepare data loader for evaluation"""
        dataset = LaMPDataset(
            json_path=json_path,
            llm_tokenizer=self.llm_tokenizer,
            beh_tokenizer=self.beh_tokenizer
        )
        
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
    
    def _evaluate_task(self, val_loader, metric_name):
        """Evaluate model on a single task"""
        all_preds, all_labels = [], []

        with self.accelerator.autocast():
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating", 
                                 disable=not self.accelerator.is_local_main_process):
                    generated_ids = self.model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        beh_input_ids=batch["beh_input_ids"],
                        beh_attention_mask=batch["beh_attention_mask"],
                        max_length=150,
                        temperature=0.7,
                        num_beams=5,
                        early_stopping=True
                    )
                    
                    # Синхронизация
                    generated_ids = self.accelerator.gather(
                        self.accelerator.pad_across_processes(generated_ids, dim=1)
                    )
                    labels = self.accelerator.gather(
                        self.accelerator.pad_across_processes(batch["labels"], dim=1)
                    )
                    
                    # Только на главном процессе
                    if self.accelerator.is_local_main_process:
                        preds_text = self.llm_tokenizer.batch_decode(
                            generated_ids.cpu(), 
                            skip_special_tokens=True
                        )
                        labels_text = self.llm_tokenizer.batch_decode(
                            labels.cpu(),
                            skip_special_tokens=True
                        )
                        all_preds.extend(preds_text)
                        all_labels.extend(labels_text)
            
        if self.accelerator.is_local_main_process:
            return compute_metric(metric_name, all_preds, all_labels, self.llm_tokenizer)
        return None