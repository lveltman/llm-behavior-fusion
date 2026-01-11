import json
import random
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split

# === 1. Загрузка данных ===
main_path = "data/LaMP_1/lamp1_author_relevance_first_author_filtered.json"
extra_path = "data/LaMP_1/train.json"
# main_path = "data/LaMP_1/lamp1_author_relevance_first_author_filtered.json"
# extra_path = "data/LaMP_1/train_last_triplets.json"

# main_path = "data/LaMP_1/kaggle_titles.json"
# extra_path = "data/LaMP_1/train_all_titles.json"

with open(main_path, "r", encoding="utf-8") as f:
    main_data = json.load(f)
with open(extra_path, "r", encoding="utf-8") as f:
    extra_data = json.load(f)

print(f"Основной датасет: {len(main_data)} примеров")
print(f"Доп. датасет:      {len(extra_data)} примеров")

N = len(extra_data)

# === 2. Группировка основного датасета по авторам ===
author_to_examples = defaultdict(list)
for ex in main_data:  # ← было: main_ → должно быть: main_data
    author_to_examples[ex["id"]].append(ex)

# === 3. Выбор авторов для удаления (N примеров) ===
# Собираем список (author_id, список_примеров)
authors_examples = list(author_to_examples.items())
random.seed(42)
random.shuffle(authors_examples)

# Выбираем примеры до тех пор, пока не наберём >= N
to_remove = []
removed_count = 0
i = 0
while removed_count < N and i < len(authors_examples):
    author_id, examples = authors_examples[i]
    # Удалим ВСЕ примеры автора (чтобы не нарушить author-disjoint)
    if removed_count + len(examples) <= N:
        to_remove.append(author_id)
        removed_count += len(examples)
    i += 1

# Если не хватает — удалим часть примеров одного автора (осторожно!)
if removed_count < N and i < len(authors_examples):
    author_id, examples = authors_examples[i]
    needed = N - removed_count
    # Удалим только `needed` примеров этого автора
    # → Но это нарушит author-disjoint, если автор останется в основном датасете!
    # Поэтому лучше НЕ делать так. Вместо этого — удалим ВСЕГО автора, даже если превысим N.
    to_remove.append(author_id)
    removed_count += len(examples)

print(f"Удаляем {removed_count} примеров от {len(to_remove)} авторов")

# === 4. Формирование обновлённого основного датасета ===
main_data_pruned = []
for author_id, examples in author_to_examples.items():
    if author_id not in to_remove:
        main_data_pruned.extend(examples)

# === 4.1. Удаляем из основного датасета авторов, которые есть в extra_data ===
extra_authors = set(ex["id"] for ex in extra_data)
main_data_pruned = [ex for ex in main_data_pruned if ex["id"] not in extra_authors]

print(f"После удаления пересекающихся авторов: {len(main_data_pruned)} примеров")

# === 5. Объединение ===
combined_train = main_data_pruned + extra_data
print(f"Итоговый train: {len(combined_train)} примеров (было {len(main_data)})")


# === 6. Разделение на train/val (только по оставшимся авторам из основного) ===
# ВАЖНО: второй датасет НЕ участвует в разделении — он весь идёт в train

# Группируем только оставшихся авторов из основного
author_to_examples_pruned = defaultdict(list)
for ex in main_data_pruned:
    author_to_examples_pruned[ex["id"]].append(ex)

# Стратификация (как раньше)
author_labels = {}
for author_id, examples in author_to_examples_pruned.items():
    y = [1 if ex["output_text"] == "[1]" else 0 for ex in examples]
    author_labels[author_id] = sum(y) / len(y)

authors = list(author_labels.keys())
if len(authors) == 0:
    raise ValueError("После удаления не осталось авторов для разделения!")

y_authors = [author_labels[aid] for aid in authors]
bins = np.linspace(0, 1, num=6)
stratify_classes = np.digitize(y_authors, bins) - 1
stratify_classes = np.clip(stratify_classes, 0, len(bins) - 2)

train_authors, val_authors = train_test_split(
    authors,
    test_size=0.2,
    random_state=42,
    stratify=stratify_classes
)
train_authors = set(train_authors)
val_authors = set(val_authors)

# Формируем train и val
train_final = []
val_final = []

# Примеры из основного датасета
for author_id, examples in author_to_examples_pruned.items():
    if author_id in train_authors:
        train_final.extend(examples)
    else:
        val_final.extend(examples)

# Добавляем ВЕСЬ доп. датасет в train
train_final.extend(extra_data)

# === 7. Статистика ===
def count_labels(data):
    from collections import Counter
    return dict(Counter(ex["output_text"] for ex in data))

print(f"\n✅ Итоги:")
print(f"Train: {len(train_final)} примеров (включая все {len(extra_data)} из доп. датасета)")
print(f"Val:   {len(val_final)} примеров (только из основного)")
print(f"Распределение меток в train: {count_labels(train_final)}")
print(f"Распределение меток в val:   {count_labels(val_final)}")

# === 8. Сохранение ===
path_train = "data/LaMP_1/train_kaggle_lamp_all.json"
path_val = "data/LaMP_1/val_kaggle_lamp_all.json"

# path_train = "data/LaMP_1/train_kaggle_lamp_last_triplets.json"
# path_val = "data/LaMP_1/val_kaggle_lamp_last_triplets.json"
# val_kaggle_lamp_last_triplets == val_kaggle_lamp_all

# path_train = "data/LaMP_1/train_kaggle_titles_lamp_titles.json"
# path_val = "data/LaMP_1/val_kaggle_titles.json"

with open(path_train, "w", encoding="utf-8") as f:
    json.dump(train_final, f, ensure_ascii=False, indent=2)

with open(path_val, "w", encoding="utf-8") as f:
    json.dump(val_final, f, ensure_ascii=False, indent=2)

print("\n✅ Сохранено!")