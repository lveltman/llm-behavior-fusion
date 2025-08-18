import torch
from accelerate import Accelerator
from transformers import AutoTokenizer

from model import BehavioralEncoder, QFormer, FusionModel, BehavioralTwin
from trainers.trainer import TaskSequentialTrainer, JointTaskTrainer, ModelEvaluator, ModelSaver


if __name__ == "__main__":
    from accelerate import Accelerator
    import torch
    from transformers import AutoTokenizer
    
    accelerator = Accelerator()
    
    mode = "joint"       # "joint", "sequential", "eval_only"
    resume_from = None
    num_epochs = 10
    batch_size = 24
    lr = 1e-4
    warmup_ratio = 0.1   # warmup 10% шагов
    
    beh_enc_name = "BAAI/bge-base-en-v1.5"
    llm_name = "google/flan-t5-xl"
    dim = 768
    
    tasks = {
        "LaMP-1": {"json_path": "data/LaMP_1/train.json", "metric": "accuracy"},
        "LaMP-3": {"json_path": "data/LaMP_3/train.json", "metric": "regression"},
        "LaMP-4": {"json_path": "data/LaMP_4/train.json", "metric": "rouge"},
        "LaMP-5": {"json_path": "data/LaMP_5/train.json", "metric": "rouge"},
        "LaMP-7": {"json_path": "data/LaMP_7/train.json", "metric": "rouge"},
    }
    
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    beh_tokenizer = AutoTokenizer.from_pretrained(beh_enc_name)
    
    model = BehavioralTwin(
        BehavioralEncoder(encoder_name=beh_enc_name),
        QFormer(hidden_size=dim),
        FusionModel(llm_name=llm_name, beh_hidden_size=dim)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    if resume_from:
        accelerator.print(f"Loading checkpoint from {resume_from}")
        state_dict = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(state_dict)
    
    if mode == "joint":
        # prepare model+optimizer+dataloader
        trainer = JointTaskTrainer(accelerator, model, optimizer, llm_tokenizer, beh_tokenizer)
        model = trainer.train("data/lamp_all_train.json", num_epochs, batch_size, warmup_ratio)
    
    elif mode == "sequential":
        trainer = TaskSequentialTrainer(accelerator, model, optimizer, llm_tokenizer, beh_tokenizer)
        model = trainer.train(tasks, num_epochs, batch_size, warmup_ratio)
    
    elif mode == "eval_only":
        accelerator.print("Evaluation only mode — skipping training.")
        model.to(accelerator.device)
    else:
        raise ValueError(f"Unknown mode {mode}")
    
    accelerator.print("Running final evaluation...")
    model.eval()
    evaluator = ModelEvaluator(accelerator, model, llm_tokenizer, beh_tokenizer)
    final_metrics = evaluator.evaluate(tasks)
    accelerator.print(f"Final metrics: {final_metrics}")
