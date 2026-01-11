import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer
from bitsandbytes.optim import AdamW8bit
import gc
from time import time
from model import BehavioralEncoder, QFormer, FusionModel, BehavioralTwin, MemoryOptimizedBehavioralTwin, SimpleTextEncoder
from trainers.trainer import TaskSequentialTrainer, JointTaskTrainer, ModelEvaluator, ModelSaver
from pplug import PPlugModel

if __name__ == "__main__":

    accelerator = Accelerator(
        mixed_precision='bf16',  # или 'fp16'
        device_placement=True,
        gradient_accumulation_steps=1,
        cpu=False
    )
    
    mode = "sequential"       # "joint", "sequential", "eval_only"
    resume_from = "saved/checkpoints/bge_m3_xxlflan_new_qformer1_20251208_182256.pt" 
    resume_from = "saved/checkpoints/pplug_LaMP-1_epoch_2_step_69471_20251209_180048.pt" 
    resume_from = "saved/checkpoints/pplug_LaMP-1_epoch_1_step_80000_20251211_070836.pt"
    resume_from = "saved/checkpoints/llm_beh_kaggle_lamp_LaMP-1_epoch_3_step_240000_20251214_052017.pt"
    resume_from = "saved/checkpoints/llm_beh_kaggle_lamp_all_LaMP-1_epoch_3_step_240000_20251217_021144.pt" 
    resume_from = "saved/checkpoints/llm_beh_kaggle_lamp_titles_LaMP-1_epoch_3_20251218_205942.pt"#"saved/checkpoints/model_LaMP-1_epoch10_20250830_220244.pt"
    resume_from = ""
    # resume_from = "saved/checkpoints/optuna_trial_1_LaMP-1_epoch2_20251129_064651.pt"
    # resume_from = "saved/checkpoints/bge_base_flan_t5_xxl_new_qformer_LaMP-1_epoch3_20251005_203544.pt"
    num_epochs = 3
    batch_size = 4

    # lr = 0.00025237006987168306
    # warmup_ratio = 0.07331269248091292
    # num_queries = 8

    # lr = 3.4324099556985233e-06
    # warmup_ratio = 0.13659563587927478
    # weight_decay = 0.00020176083553103118

    lr = 1e-4
    warmup_ratio = 0.05
    weight_decay = 0.001
    num_queries = 8
    
    beh_enc_name = "BAAI/bge-base-en-v1.5"
    # beh_enc_name = "facebook/contriever"
    # beh_enc_name = "Alibaba-NLP/gte-large-en-v1.5"
    # beh_enc_name = "BAAI/bge-m3"
    # llm_name = "google/flan-t5-xl"
    llm_name = "google/flan-t5-xxl"
    # llm_name = "Qwen/Qwen2-7B"
    # llm_name = "Qwen/Qwen3-14B"

    # warmup_ratio = 0.053494464405939135, 'num_queries': 11, 'weight_decay': 0.0010468580083440608, 'lr': 0.00048789888760538466, 'beta1': 0.9165027242838241, 'beta2': 0.9412469564025118}
    

    # exp_name = "qwen14b"
    exp_name = "pplug_lamp_all_lamp"
    # exp_name = "llm_qwen_beh_lamp_all_titles"
    
    max_len_beh = 512 if "base" in beh_enc_name else 1024
    llm_type = "decoder-only" if 'qwen' in llm_name.lower() else "encoder-decoder"
    max_len_llm = 2048 if llm_type == "decoder-only" else 512 # or 1024 for flan
    max_len_llm = 512
    print(llm_type)
    print(max_len_llm)
    
    use_memory_optimization = False
    tasks = {
         "LaMP-1": {"train_json_path": "data/LaMP_1/train_kaggle_lamp_all.json",
                   "val_json_path":  "data/LaMP_1/val_kaggle_lamp_all.json", 
                    "metric": "accuracy"},
    }
    # tasks = {
    #     # "LaMP-1": {"train_json_path": "data/LaMP_1/lamp1_author_relevance_first_author_filtered.json",
    #     #            "val_json_path": "data/LaMP_1/dev.json",
    #     #            "metric": "accuracy"},
    #     # "LaMP-1": {"train_json_path": "data/LaMP_1/train_kaggle.json",
    #     #            "val_json_path": "data/LaMP_1/val_kaggle.json",
    #     #            "metric": "accuracy"},
    #     "LaMP-1": {"train_json_path": "data/LaMP_1/train_combined_1.json",
    #                "val_json_path":  "data/LaMP_1/val_combined_1.json", #dev_last_triplets.json",#"data/LaMP_1/val_combined_1.json",
    #                "metric": "accuracy"},
    #     # "LaMP-1": {"train_json_path": "data/LaMP_1/train_kaggle_titles_lamp_titles.json",
    #     #            "val_json_path":  "data/LaMP_1/val_kaggle_titles.json",  #"data/LaMP_1/train_all_titles.json", #
    #     #            "metric": "accuracy"},
    #     # "LaMP-1": {"train_json_path": "data/LaMP_1/train_all_titles.json",
    #     #    "val_json_path":  "data/LaMP_1/dev_all_titles.json", #"data/LaMP_1/val_kaggle_titles.json", 
    #     #    "metric": "accuracy"},
    #     # "LaMP-3": {"json_path": "data/LaMP_3/train.json", "metric": "regression"},
    #     # "LaMP-4": {"json_path": "data/LaMP_4/train.json", "metric": "rouge"},
    #     # "LaMP-5": {"json_path": "data/LaMP_5/train.json", "metric": "rouge"},
    #     # "LaMP-7": {"json_path": "data/LaMP_7/train.json", "metric": "rouge"},
    # }
    
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    beh_tokenizer = AutoTokenizer.from_pretrained(beh_enc_name)

    print(f"self.llm_tokenizer.pad_token_id = {llm_tokenizer.pad_token_id}")
    print(f"self.beh_tokenizer.pad_token_id = {beh_tokenizer.pad_token_id}")

    beh_encoder = BehavioralEncoder(beh_enc_name)
    input_encoder = SimpleTextEncoder(beh_enc_name)
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

        # model = PPlugModel(beh_encoder_name="BAAI/bge-base-en-v1.5",
        #             input_encoder_name="BAAI/bge-base-en-v1.5",
        #             llm_name=llm_name)

    model.train()
    
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # print("input_encoder:", count_params(model.input_encoder))
    # print("qformer:", count_params(model.qformer))
    # print("prefix_proj:", count_params(model.fusion.prefix_proj))
    # print("instruction:", count_params(model.fusion.instruction_processor) + model.fusion.instruction_embedding.numel())
    # print("instruction:", model.fusion.instruction_embedding.numel())

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    if resume_from:
        accelerator.print(f"Loading checkpoint from {resume_from}")
        
        try:
            # Вариант 1: Загрузка на CPU
            checkpoint = torch.load(resume_from, map_location='cpu')
            accelerator.print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
            
            # Вариант 2: Попиксельное загрузка для больших моделей
            def load_checkpoint_safely(checkpoint_path, model):
                """Безопасная загрузка чекпоинта"""
                # Сначала загрузите на CPU
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # Если это dict
                if isinstance(checkpoint, dict):
                    # Пробуем разные ключи
                    possible_keys = ['model_state_dict', 'state_dict', 'model']
                    for key in possible_keys:
                        if key in checkpoint:
                            state_dict = checkpoint[key]
                            break
                    else:
                        # Если не нашли ключей, возможно это сам state_dict
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # Фильтрация ключей (если архитектура изменилась)
                model_state_dict = model.state_dict()
                
                # Только совпадающие ключи
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                      if k in model_state_dict and v.shape == model_state_dict[k].shape}
                
                accelerator.print(f"Loaded {len(filtered_state_dict)}/{len(state_dict)} layers")
                
                # Загружаем
                model.load_state_dict(filtered_state_dict, strict=False)
                
                # Очистка
                del checkpoint, state_dict, filtered_state_dict
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Используем безопасную загрузку
            load_checkpoint_safely(resume_from, model)
            
        except Exception as e:
            accelerator.print(f"Error loading checkpoint: {e}")
            accelerator.print("Continuing with random initialization")
    
    accelerator.print(f"Memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    
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
