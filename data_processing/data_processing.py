import os
import requests
import json

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"{save_path} already exists, skipping download.")
        return
    print(f"Downloading {url} ...")
    r = requests.get(url)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)
    print(f"Saved to {save_path}")


def normalize_profile(profile_item):
    parts = []
    if "title" in profile_item and profile_item["title"]:
        parts.append(f"""TITLE: \"{profile_item['title']}\"""")
    # if "abstract" in profile_item and profile_item["abstract"]:
    #     parts.append(f"ABSTRACT: {profile_item['abstract']}")
    return " ".join(parts)



def preprocess_all_lamp_with_first_n_profiles(datasets, base_dir="data", split="train", save_path="data/lamp_all_train.json", num_profiles=6):
    all_data = []

    for dataset_id in datasets:
        questions_path = os.path.join(base_dir, f"LaMP_{dataset_id}", f"{split}", f"{split}_questions.json")
        outputs_path = os.path.join(base_dir, f"LaMP_{dataset_id}", f"{split}", f"{split}_outputs.json")

        with open(questions_path, "r") as f:
            questions = json.load(f)

        with open(outputs_path, "r") as f:
            outputs_data = json.load(f)
            outputs_map = {o["id"]: o["output"] for o in outputs_data["golds"]}

        for q in questions:
            q_id = q["id"]
            if q_id not in outputs_map:
                continue

            profiles_norm = []
            for i, profile in enumerate(q["profile"]):
                if i == num_profiles:
                    break
                norm_profile = normalize_profile(profile)
                profiles_norm.append(norm_profile)
            # profiles_norm = [normalize_profile(p) for p in q["profile"]]
            # scores = [int(p["score"]) if "score" in p and p["score"] not in (None, "") else None for p in q["profile"]]

            all_data.append({
                "task": f"LaMP_{dataset_id}",
                "id": q_id,
                "input_text": q["input"],
                "output_text": outputs_map[q_id],
                "behavior_profile_text": profiles_norm,
                # "behavior_scores": scores
            })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Saved combined dataset to {save_path}, total {len(all_data)} examples")


def preprocess_all_lamp_with_last_triplets(
    datasets,
    base_dir="data",
    split="train",
    save_path="data/lamp_all_train.json",
    titles_per_sample=3
):
    """
    Генерация одного семпла на автора:
      - берём последние `titles_per_sample` публикаций из профиля автора
      - если публикаций меньше — берём все
    """
    all_data = []

    for dataset_id in datasets:
        questions_path = os.path.join(base_dir, f"LaMP_{dataset_id}", split, f"{split}_questions.json")
        outputs_path = os.path.join(base_dir, f"LaMP_{dataset_id}", split, f"{split}_outputs.json")

        with open(questions_path, "r") as f:
            questions = json.load(f)

        with open(outputs_path, "r") as f:
            outputs_data = json.load(f)
            outputs_map = {o["id"]: o["output"] for o in outputs_data["golds"]}

        for q in questions:
            q_id = q["id"]
            if q_id not in outputs_map:
                continue

            # Извлекаем и нормализуем только те публикации, у которых есть title
            titles = [
                normalize_profile(p) 
                for p in q["profile"] 
                if p.get("title")
            ]

            if not titles:
                continue

            # Берём последние N статей (а не все или блоки)
            last_titles = titles[-titles_per_sample:]  # последние titles_per_sample

            all_data.append({
                "task": f"LaMP_{dataset_id}",
                "id": q_id,
                "input_text": q["input"],
                "output_text": outputs_map[q_id],
                "behavior_profile_text": last_titles,  # список из 1–3 строк
            })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Saved dataset to {save_path}, total {len(all_data)} examples.")


def preprocess_all_lamp_with_fixed_triplets(
    datasets,
    base_dir="data",
    split="train",
    save_path="data/lamp_all_train.json",
    titles_per_sample=3
):
    """
    Генерация семплов для каждого автора:
      - все публикации автора берутся
      - делим на последовательные блоки по titles_per_sample
      - если в конце меньше, чем titles_per_sample, используем оставшиеся
    """
    all_data = []

    for dataset_id in datasets:
        questions_path = os.path.join(base_dir, f"LaMP_{dataset_id}", f"{split}", f"{split}_questions.json")
        outputs_path = os.path.join(base_dir, f"LaMP_{dataset_id}", f"{split}", f"{split}_outputs.json")

        with open(questions_path, "r") as f:
            questions = json.load(f)

        with open(outputs_path, "r") as f:
            outputs_data = json.load(f)
            outputs_map = {o["id"]: o["output"] for o in outputs_data["golds"]}

        for q in questions:
            q_id = q["id"]
            if q_id not in outputs_map:
                continue

            titles = [normalize_profile(p) for p in q["profile"] if p.get("title")]
            if not titles:
                continue  # если вообще нет публикаций, пропускаем

            # Разбиваем на последовательные блоки по titles_per_sample
            for i in range(0, len(titles), titles_per_sample):
                sample_titles = titles[i:i+titles_per_sample]
                # sample_titles = " ".join(sample_titles)
            
                all_data.append({
                    "task": f"LaMP_{dataset_id}",
                    "id": q_id,
                    "input_text": q["input"],
                    "output_text": outputs_map[q_id],
                    # "behavior_profile_text": 'titles: ' + sample_titles,
                    "behavior_profile_text": sample_titles,
                })

                
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"Saved dataset to {save_path}, total {len(all_data)} examples.")



datasets = [1, 3, 4, 5, 7]
datasets = [1]
for dataset in datasets:
    # https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_1/train/
    base_url = f"https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_{dataset}/train/"
    base_path = f"data/LaMP_{dataset}"
    train_path = os.path.join(base_path, "train")
    dev_path = os.path.join(base_path, "dev")
    
    print(f"Creating directories for dataset {dataset}:")
    print("  ->", train_path)
    print("  ->", dev_path)
    
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(dev_path, exist_ok=True)
    
    download_file(base_url + "train_questions.json", os.path.join(train_path, "train_questions.json"))
    download_file(base_url + "train_outputs.json", os.path.join(train_path, "train_outputs.json"))

    base_url = f"https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_{dataset}/dev/"
    download_file(base_url + "dev_questions.json", os.path.join(dev_path, "dev_questions.json"))
    download_file(base_url + "dev_outputs.json", os.path.join(dev_path, "dev_outputs.json"))


dataset_ids = [1]

for dataset_id in dataset_ids:
    # preprocess_all_lamp_with_fixed_triplets(
    #     datasets=[dataset_id],
    #     base_dir="data",
    #     split="train",
    #     save_path=f"data/LaMP_{dataset_id}/train_all_titles.json",
    #     titles_per_sample=300
    # )

    preprocess_all_lamp_with_fixed_triplets(
        datasets=[dataset_id],
        base_dir="data",
        split="dev",
        save_path=f"data/LaMP_{dataset_id}/dev_all_titles_300.json",
        titles_per_sample=300
    )
