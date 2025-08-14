import torch
from accelerate import Accelerator
from transformers import AutoTokenizer

from model import BehavioralEncoder, QFormer, FusionModel, BehavioralTwin
from trainers.trainer import TaskSequentialTrainer, JointTaskTrainer, ModelEvaluator


if __name__ == "__main__":
    # Initialize accelerator and model components
    accelerator = Accelerator()

    # Choose training mode
    mode = "sequential"  # or "joint"
    # Model configuration
    beh_enc_name = "BAAI/bge-base-en-v1.5"
    llm_name = "google/flan-t5-base"
    dim = 768
    
    # Training configuration
    num_epochs = 5
    batch_size = 100
    
    # Task definitions
    tasks = {
        "LaMP-1": {
            "json_path": "data/LaMP_1/train.json",
            "metric": "accuracy"
        },
        "LaMP-3": {
            "json_path": "data/LaMP_3/train.json",
            "metric": "regression"
        },
        "LaMP-4": {
            "json_path": "data/LaMP_4/train.json",
            "metric": "rouge"
        },
        "LaMP-5": {
            "json_path": "data/LaMP_5/train.json",
            "metric": "rouge"
        },
        "LaMP-7": {
            "json_path": "data/LaMP_7/train.json",
            "metric": "rouge"
        },
    }
    
    # Initialize tokenizers and model
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    beh_tokenizer = AutoTokenizer.from_pretrained(beh_enc_name)
    
    model = BehavioralTwin(
        BehavioralEncoder(encoder_name=beh_enc_name),
        QFormer(hidden_size=dim),
        FusionModel(llm_name=llm_name, beh_hidden_size=dim)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    if mode == "sequential":
        trainer = TaskSequentialTrainer(
            accelerator, model, optimizer, llm_tokenizer, beh_tokenizer
        )
        model = trainer.train(tasks, num_epochs, batch_size)
    else:
        trainer = JointTaskTrainer(
            accelerator, model, optimizer, llm_tokenizer, beh_tokenizer
        )
        model = trainer.train("data/lamp_all_train.json", num_epochs, batch_size)
    
    # Evaluate final model
    evaluator = ModelEvaluator(accelerator, model, llm_tokenizer, beh_tokenizer)
    final_metrics = evaluator.evaluate(tasks)

    ModelSaver(accelerator).save_model(model, "final")