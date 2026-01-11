import json

# Укажи пути
input_path = "/home/veltman.lina/llm-behavior-fusion/data/LaMP_1/lamp1_author_relevance_first_author_.json"
output_path = "/home/veltman.lina/llm-behavior-fusion/data/LaMP_1/lamp1_author_relevance_first_author_filtered.json"

print(f"Загрузка {input_path}...")
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Исходное количество примеров: {len(data)}")

# Фильтрация: оставить только те, у кого behavior_profile_text — непустой список
filtered_data = [
    item for item in data
    if isinstance(item.get("behavior_profile_text"), list) and len(item["behavior_profile_text"]) > 0
]

print(f"После фильтрации: {len(filtered_data)} примеров")
print(f"Удалено: {len(data) - len(filtered_data)} примеров с пустым behavior_profile_text")

# Сохранение
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

print(f"✅ Сохранено в {output_path}")