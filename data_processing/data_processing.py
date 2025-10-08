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
        parts.append(f"TITLE: {profile_item['title']}")
    if "abstract" in profile_item and profile_item["abstract"]:
        parts.append(f"ABSTRACT: {profile_item['abstract']}")
    # if "text" in profile_item and profile_item["text"]:
    #     parts.append(f"TEXT: {profile_item['text']}")
    # if "score" in profile_item and profile_item["score"] not in (None, ""):
    #     parts.append(f"SCORE: {profile_item['score']}")
    # if "date" in profile_item and profile_item["date"]:
    #     parts.append(f"DATE: {profile_item['date']}")
    # if "id" in profile_item and profile_item["id"]:
    #     parts.append(f"ID: {profile_item['id']}")
    return " ".join(parts)

def preprocess_all_lamp(datasets, base_dir="data", split="train", save_path="data/lamp_all_train.json", num_profiles=6):
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



datasets = [1, 3, 4, 5, 7]
datasets = [1]
for dataset in datasets:
    base_url = f"https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_{dataset}/train/"
    os.makedirs(f"..data/LaMP_{dataset}/train", exist_ok=True)
    os.makedirs(f"../data/LaMP_{dataset}/dev", exist_ok=True)
    
    download_file(base_url + "train_questions.json", f"../data/LaMP_{dataset}/train/train_questions.json")
    download_file(base_url + "train_outputs.json", f"../data/LaMP_{dataset}/train/train_outputs.json")

    base_url = f"https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_{dataset}/dev/"
    download_file(base_url + "dev_questions.json", f"../data/LaMP_{dataset}/dev/dev_questions.json")
    download_file(base_url + "dev_outputs.json", f"../data/LaMP_{dataset}/dev/dev_outputs.json")


dataset_ids = [1, 3, 4, 5, 7]

dataset_ids = [1]

for dataset_id in dataset_ids:
    preprocess_all_lamp(
        datasets=[dataset_id],
        base_dir="../data",
        split="train",
        save_path=f"../data/LaMP_{dataset_id}/train.json",
        num_profiles=16
    )
    preprocess_all_lamp(
        datasets=[dataset_id],
        base_dir="../data",
        split="dev",
        save_path=f"../data/LaMP_{dataset_id}/dev.json",
        num_profiles=16
    )

