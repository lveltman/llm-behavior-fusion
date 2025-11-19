import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import optuna
from accelerate import Accelerator
from transformers import AutoTokenizer


from model import BehavioralEncoder, SimpleTextEncoder, QFormer, FusionModel, BehavioralTwin, MemoryOptimizedBehavioralTwin
from trainers.trainer import TaskSequentialTrainer, JointTaskTrainer, ModelEvaluator, ModelSaver

def objective(trial):
    # Оптимизация гиперпараметров
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    batch_size = 2 #trial.suggest_categorical("batch_size", [2, 4, 8, 16])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.2, log=True)
    num_queries = 8
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    
    # Оптимизация архитектурных параметров
    max_len_beh = 512
    max_len_llm = 512
    
    # Выбор модели (можно раскомментировать для оптимизации архитектуры)
    # beh_enc_name = trial.suggest_categorical("beh_enc_name", [
    #     "BAAI/bge-base-en-v1.5",
    #     "facebook/contriever", 
    #     "Alibaba-NLP/gte-large-en-v1.5"
    # ])
    
    # beh_enc_name = "BAAI/bge-base-en-v1.5"
    # llm_name = "google/flan-t5-xxl"
        # beh_enc_name = "BAAI/bge-base-en-v1.5"
    # beh_enc_name = "facebook/contriever"
    # beh_enc_name = "Alibaba-NLP/gte-large-en-v1.5"
    beh_enc_name = "BAAI/bge-m3"
    # llm_name = "google/flan-t5-xl"
    llm_name = "google/flan-t5-xxl"
    # llm_name = "Qwen/Qwen2-7B"
    
    # Фиксированные параметры для ускорения оптимизации
    mode = "sequential"
    num_epochs = 2  # Меньше эпох для быстрой оптимизации
    use_memory_optimization = False
    exp_name = f"optuna_trial_{trial.number}"

    accelerator = Accelerator()
    
    # Определяем тип LLM и максимальную длину
    llm_type = "decoder-only" if 'qwen' in llm_name.lower() else "encoder-decoder"
    
    tasks = {
        "LaMP-1": {"json_path": "data/LaMP_1/train.json", "metric": "accuracy"},
        # "LaMP-3": {"json_path": "data/LaMP_3/train.json", "metric": "regression"},
    }
    
    try:
        # Инициализация компонентов модели
        llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
        beh_tokenizer = AutoTokenizer.from_pretrained(beh_enc_name)

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

        model.train()
        
        # Оптимизатор с оптимизированными параметрами
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        
        # Обучение
        trainer = TaskSequentialTrainer(
            accelerator, model, optimizer,
            llm_tokenizer, beh_tokenizer, beh_tokenizer,
            max_len_llm, max_len_beh, llm_type, exp_name
        )
        
        model = trainer.train(tasks, num_epochs, batch_size, warmup_ratio)
        
        # Оценка модели
        model.eval()
        evaluator = ModelEvaluator(
            accelerator, model,
            llm_tokenizer, beh_tokenizer, beh_tokenizer,
            max_len_llm, max_len_beh, llm_type, exp_name
        )
        
        final_metrics = evaluator.evaluate(tasks)
        
        # Возвращаем значение метрики для минимизации (например, отрицательную accuracy)
        # Или комбинированную метрику для нескольких задач
        main_metric = list(final_metrics.values())[0].get("accuracy", 0)
        return -main_metric  # Optuna минимизирует, поэтому возвращаем отрицательное значение
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('inf')  # Возвращаем большое значение для неудачных trials

def run_optuna_optimization():
    """Запуск оптимизации с Optuna"""
    
    study = optuna.create_study(
        direction="minimize",  # Мы минимизируем отрицательную accuracy
        study_name="behavioral_twin_optimization",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True  # Продолжить существующее исследование
    )
    
    # Запуск оптимизации
    study.optimize(objective, n_trials=15, timeout=48*3600)  # 24 часа максимум
    
    # Вывод результатов
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {-trial.value}")  # Преобразуем обратно в положительное значение
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Сохранение лучших параметров
    best_params = trial.params
    print("\nBest parameters for your script:")
    print(f"lr = {best_params.get('lr')}")
    print(f"batch_size = {best_params.get('batch_size')}")
    print(f"warmup_ratio = {best_params.get('warmup_ratio')}")
    print(f"num_queries = {best_params.get('num_queries')}")
    print(f"weight_decay = {best_params.get('weight_decay')}")
    print(f"max_len_beh = {best_params.get('max_len_beh')}")
    print(f"max_len_llm = {best_params.get('max_len_llm')}")
    
    return study

def run_final_training(best_params):
    """Запуск финального обучения с лучшими параметрами"""
    
    accelerator = Accelerator()
    
    mode = "sequential"
    num_epochs = 10  # Больше эпох для финального обучения
    resume_from = None
    
    # Используем лучшие параметры из Optuna
    lr = best_params.get("lr", 0.00025237006987168306)
    batch_size = best_params.get("batch_size", 2)
    warmup_ratio = best_params.get("warmup_ratio", 0.07331269248091292)
    num_queries = best_params.get("num_queries", 8)
    weight_decay = best_params.get("weight_decay", 0.01)
    max_len_beh = best_params.get("max_len_beh", 512)
    max_len_llm = best_params.get("max_len_llm", 512)

    beh_enc_name = "BAAI/bge-base-en-v1.5"
    llm_name = "google/flan-t5-xxl"
    
    exp_name = "optuna_optimized_final"
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
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    
    if resume_from:
        accelerator.print(f"Loading checkpoint from {resume_from}")
        state_dict = torch.load(resume_from)
        model.load_state_dict(state_dict)
    
    # Обучение с лучшими параметрами
    trainer = TaskSequentialTrainer(
        accelerator, model, optimizer,
        llm_tokenizer, beh_tokenizer, beh_tokenizer,
        max_len_llm, max_len_beh, "encoder-decoder", exp_name
    )
    
    model = trainer.train(tasks, num_epochs, batch_size, warmup_ratio)
    
    # Финальная оценка
    accelerator.print("Running final evaluation...")
    model.eval()
    evaluator = ModelEvaluator(
        accelerator, model,
        llm_tokenizer, beh_tokenizer, beh_tokenizer,
        max_len_llm, max_len_beh, "encoder-decoder", exp_name
    )
    
    final_metrics = evaluator.evaluate(tasks)
    accelerator.print(f"Final metrics: {final_metrics}")
    
    return model, final_metrics

if __name__ == "__main__":
    print("Starting Optuna optimization...")
    study = run_optuna_optimization()
    
    # Сохраняем исследование
    import joblib
    joblib.dump(study, "optuna_study.pkl")
    print("Study saved to optuna_study.pkl")
        
    # else:
    #     # Загружаем лучшее исследование и запускаем финальное обучение
    #     try:
    #         import joblib
    #         study = joblib.load("optuna_study.pkl")
    #         best_params = study.best_trial.params
    #         print("Loaded best parameters from saved study")
    #     except:
    #         print("No saved study found, using default parameters")
    #         best_params = {}
        
    #     print("Starting final training with optimized parameters...")
    #     model, metrics = run_final_training(best_params)