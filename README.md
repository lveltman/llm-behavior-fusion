# BehavioralTwin: Q-Former Based Personalization of LLMs

> **Beyond PPlug**: We replace simple weighted averaging with a **Q-Former cross-attention module** to better model user behavior for LLM personalization.

## 🔍 Что делает проект

Проект реализует и оценивает **BehavioralTwin** — архитектуру персонализации LLM, которая:
- Кодирует историю пользователя через **замороженный BGE энкодер**
- Агрегирует поведение и запрос через **Q-Former** (learnable queries + кросс-внимание)
- Интегрирует персональный префикс в **Flan-T5-XXL / Qwen** без дообучения самой LLM

**Результат**: **+5.05% accuracy** на LaMP-1 по сравнению с PPlug (0.8997 vs 0.8492).

## 🧠 Архитектура

```
[User History] → Behavioral Encoder → [P×768]
                                  ↘
                                    ┌───────────────┐
                                    │   Q-FORMER    │ ← 8 обучаемых queries
                                    │ • Cross-att (history) │
                                    │ • Cross-att (input)   │
                                    │ • Self-att (queries)  │
                                    └───────────────┘ → [Q×H]
                                  ↗
[Current Query] → Input Encoder  → [1×768]

[Q×H] → Proj → Prefix → LLM → [1] or [2]
```

### Ключевые компоненты
- **`BehavioralEncoder`**: замороженный BGE для истории
- **`SimpleTextEncoder`**: частично обучаемый BGE для запроса (`tune_layers=4`)
- **`QFormer`**: центральный модуль агрегации (наша новизна!)
- **`FusionModel`**: поддержка Flan-T5 (encoder-decoder) и Qwen (causal)

## ⚙️ Запуск

```bash
accelerate launch --config_file config/ds_config.yaml train.py
```

### Режимы работы (`train.py`)
```python
mode = "sequential"  # обучение отдельно по задачам (LaMP-1, LaMP-2, ...)
mode = "joint"       # обучение на объединённых данных
mode = "eval_only"   # только оценка метрик

# Для продолжения обучения:
resume_from = "saved/checkpoints/your_model.pt"

# Гиперпараметры (LaMP-1):
lr = 1e-4
warmup_ratio = 0.05
num_queries = 8
batch_size = 4  # для Flan-T5-XXL
```

## 📊 Данные

- **Основной датасет**: [LaMP Benchmark](https://lamp-benchmark.github.io/download)
- **Дополнительные данные**: [Research Papers Dataset (Kaggle)](https://www.kaggle.com/datasets/nechbamohammed/research-papers-dataset)

## 📁 Структура кода

```
├── model.py             # BehavioralTwin, QFormer, FusionModel
├── pplug.py             # Реализация baseline (PPlug)
├── dataset/             # LaMPDataset с author-disjoint split
├── trainers/            # TaskSequentialTrainer, ModelEvaluator
└── train.py             # Точка входа
```

## 📈 Результаты

Разница нашей модели относительно PPlug:

| Метрика | Разница (Abs. Delta) |
|----------|-----------------------|
| Accuracy | +5.046%               |
| Precision| +5.026%               |
| Recall   | +5.051%               |


→ Улучшение достигнуто за счёт **более богатой агрегации поведения через Q-Former**.
