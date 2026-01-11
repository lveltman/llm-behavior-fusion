import json
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

# Загрузка данных
input_path = "data/LaMP_1/lamp1_author_relevance_first_author_filtered.json"
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Группировка по авторам (id)
author_to_examples = defaultdict(list)
for example in data:
    author_to_examples[example["id"]].append(example)

# Для каждого автора вычислим "представительную" метку:
# Используем долю позитивных (например, "[1]") примеров у автора
author_labels = {}
for author_id, examples in author_to_examples.items():
    labels = [ex["output_text"] for ex in examples]
    # Преобразуем в числовой: "[1]" → 1, "[2]" → 0
    y = [1 if label == "[1]" else 0 for label in labels]
    # Используем долю "[1]" как признак для стратификации
    author_labels[author_id] = sum(y) / len(y)

authors = list(author_labels.keys())
y_authors = [author_labels[aid] for aid in authors]

# Разделим авторов стратифицированно по доле [1]
# sklearn требует дискретные классы, поэтому "дискретизуем" доли
# Например: 0.0–0.2 → 0, 0.2–0.4 → 1, ..., 0.8–1.0 → 4
import numpy as np
bins = np.linspace(0, 1, num=6)  # 5 бинов: [0,0.2), [0.2,0.4), ..., [0.8,1.0]
stratify_classes = np.digitize(y_authors, bins) - 1  # индексы от 0 до 4

# Убедимся, что нет выбросов
stratify_classes = np.clip(stratify_classes, 0, len(bins) - 2)

# Stratified split авторов
train_authors, test_authors = train_test_split(
    authors,
    test_size=0.2,
    random_state=42,
    stratify=stratify_classes
)
train_authors = set(train_authors)
test_authors = set(test_authors)

# Формирование выборок
train_data = []
test_data = []
for author_id, examples in author_to_examples.items():
    if author_id in train_authors:
        train_data.extend(examples)
    else:
        test_data.extend(examples)

# Подсчёт меток
def count_labels(dataset):
    counts = defaultdict(int)
    for ex in dataset:
        counts[ex["output_text"]] += 1
    return dict(counts)

print(f"Всего авторов: {len(authors)}")
print(f"Train авторов: {len(train_authors)} → примеров: {len(train_data)}")
print(f"Test авторов: {len(test_authors)} → примеров: {len(test_data)}")

print("\nРаспределение меток:")
print("Train:", count_labels(train_data))
print("Test :", count_labels(test_data))

# # Сохранение
# with open("data/LaMP_1/train_kaggle.json", "w", encoding="utf-8") as f:
#     json.dump(train_data, f, ensure_ascii=False, indent=2)

# with open("data/LaMP_1/val_kaggle.json", "w", encoding="utf-8") as f:
#     json.dump(test_data, f, ensure_ascii=False, indent=2)

print("\n✅ Train/test сохранены с балансировкой по таргету!")