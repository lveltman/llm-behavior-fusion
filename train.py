import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer
from bitsandbytes.optim import AdamW8bit

from model import BehavioralEncoder, QFormer, FusionModel, BehavioralTwin, MemoryOptimizedBehavioralTwin
from trainers.trainer import TaskSequentialTrainer, JointTaskTrainer, ModelEvaluator, ModelSaver


if __name__ == "__main__":
    
    accelerator = Accelerator()
    
    mode = "sequential"       # "joint", "sequential", "eval_only"
    resume_from = None #"saved/checkpoints/model_LaMP-1_epoch10_20250830_220244.pt"
    # resume_from = "saved/checkpoints/bge_base_flan_t5_xxl_new_qformer_LaMP-1_epoch3_20251005_203544.pt"
    num_epochs = 3
    batch_size = 2

    lr = 0.00025237006987168306
    warmup_ratio = 0.07331269248091292
    num_queries = 8

    lr = 3.4324099556985233e-06
    warmup_ratio = 0.13659563587927478
    weight_decay = 0.00020176083553103118

    # beh_enc_name = "BAAI/bge-base-en-v1.5"
    # beh_enc_name = "facebook/contriever"
    # beh_enc_name = "Alibaba-NLP/gte-large-en-v1.5"
    beh_enc_name = "BAAI/bge-m3"
    # llm_name = "google/flan-t5-xl"
    llm_name = "google/flan-t5-xxl"
    # llm_name = "Qwen/Qwen2-7B"

    # warmup_ratio = 0.053494464405939135, 'num_queries': 11, 'weight_decay': 0.0010468580083440608, 'lr': 0.00048789888760538466, 'beta1': 0.9165027242838241, 'beta2': 0.9412469564025118}
    

    exp_name = "bge_m3_xxlflan_new_qformer"
    
    max_len_beh = 512 if "base" in beh_enc_name else 1024
    llm_type = "decoder-only" if 'qwen' in llm_name.lower() else "encoder-decoder"
    max_len_llm = 2048 if llm_type == "decoder-only" else 512 # or 1024 for flan

    print(llm_type)
    print(max_len_llm)
    
    use_memory_optimization = False
    
    tasks = {
        "LaMP-1": {"json_path": "data/LaMP_1/train.json", "metric": "accuracy"},
        # "LaMP-3": {"json_path": "data/LaMP_3/train.json", "metric": "regression"},
        # "LaMP-4": {"json_path": "data/LaMP_4/train.json", "metric": "rouge"},
        # "LaMP-5": {"json_path": "data/LaMP_5/train.json", "metric": "rouge"},
        # "LaMP-7": {"json_path": "data/LaMP_7/train.json", "metric": "rouge"},
    }
    
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    beh_tokenizer = AutoTokenizer.from_pretrained(beh_enc_name)

    beh_encoder = BehavioralEncoder(beh_enc_name)
    input_encoder = BehavioralEncoder(beh_enc_name)
    hidden_size = beh_encoder.text_encoder.config.hidden_size

    qformer = QFormer(hidden_size=hidden_size, num_queries=num_queries)
    fusion = FusionModel(llm_name=llm_name, beh_hidden_size=hidden_size)

    if use_memory_optimization:
        model = MemoryOptimizedBehavioralTwin(beh_encoder, input_encoder, qformer, fusion)
    else:
        model = BehavioralTwin(
            beh_encoder=beh_encoder,
            input_encoder=input_encoder,
            qformer=qformer,
            fusion=fusion
        )

    model.train()
    
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    
    if resume_from:
        accelerator.print(f"Loading checkpoint from {resume_from}")
        state_dict = torch.load(resume_from)
        model.load_state_dict(state_dict)
    
    if mode == "joint":
        # prepare model+optimizer+dataloader
        trainer = JointTaskTrainer(accelerator, model, optimizer,
                                   llm_tokenizer, beh_tokenizer, beh_tokenizer,
                                   max_len_llm, max_len_beh, llm_type, exp_name)
        model = trainer.train("data/lamp_all_train.json", num_epochs, batch_size, warmup_ratio)
    
    elif mode == "sequential":
        trainer = TaskSequentialTrainer(accelerator, model, optimizer,
                                        llm_tokenizer, beh_tokenizer, beh_tokenizer,
                                        max_len_llm, max_len_beh, llm_type, exp_name)
        model = trainer.train(tasks, num_epochs, batch_size, warmup_ratio)
    
    elif mode == "eval_only":
        accelerator.print("Evaluation only mode — skipping training.")
        model.to(accelerator.device)
    else:
        raise ValueError(f"Unknown mode {mode}")
    
    accelerator.print("Running final evaluation...")
    model.eval()
    evaluator = ModelEvaluator(accelerator, model,
                                llm_tokenizer, beh_tokenizer, beh_tokenizer,
                                max_len_llm, max_len_beh, llm_type, exp_name)
    final_metrics = evaluator.evaluate(tasks)
    accelerator.print(f"Final metrics: {final_metrics}")
