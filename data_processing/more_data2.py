from tqdm import tqdm

import pandas as pd
import numpy as np
import re
import ast
import random
from itertools import chain
import json


from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import ast
import random
from itertools import chain


import pandas as pd
import numpy as np
import ast
import random
from tqdm import tqdm
import json
from collections import defaultdict

# -------------------------------------------------
# Парсер авторов (упрощённый)
# -------------------------------------------------
def parse_authors(authors_str):
    """Парсит строку вида \"['Name1', 'Name2']\" → список строк"""
    if not isinstance(authors_str, str):
        return []
    authors_str = authors_str.strip()
    if not (authors_str.startswith("[") and authors_str.endswith("]")):
        return []
    try:
        lst = ast.literal_eval(authors_str)
        return [str(x).strip() for x in lst if x and str(x).strip() != '...']
    except:
        return []

from transformers import AutoTokenizer
beh_enc_name = "BAAI/bge-base-en-v1.5"
beh_tokenizer = AutoTokenizer.from_pretrained(beh_enc_name)

def build_dataset_fast(
    df,
    max_authors=20_000,
    max_examples_per_author=3,
    max_profile_size=10,
    total_max_examples=50_000
):
    print("🔍 Parsing authors...")
    df['authors_list'] = df['authors'].apply(parse_authors)
    df = df[df['authors_list'].apply(len) > 0].copy()
    
    # EXPLODE
    df_exploded = df.explode('authors_list').reset_index(drop=True)
    df_exploded.rename(columns={'authors_list': 'author'}, inplace=True)
    
    # Предзагрузка метаданных
    id_to_title = df_exploded.set_index('id')['title'].fillna("").to_dict()
    id_to_abstract = df_exploded.set_index('id')['abstract'].fillna("").to_dict()
    all_ids = set(id_to_title.keys())
    
    # Группировка: автор → список ID
    print("ParallelGrouping authors...")
    author_to_ids = df_exploded.groupby('author')['id'].apply(list).to_dict()
    # Фильтр: только авторы с ≥2 статьями
    author_to_ids = {a: ids for a, ids in author_to_ids.items() if len(ids) >= 2}
    
    # Ограничение числа авторов
    selected_authors = list(author_to_ids.keys())
    random.shuffle(selected_authors)
    selected_authors = selected_authors[:max_authors]
    
    print(f"✅ Processing {len(selected_authors)} authors (out of {len(author_to_ids)} total)")
    
    results = []
    total_examples = 0
    
    for author in tqdm(selected_authors, desc="Processing authors"):
        if total_max_examples and total_examples >= total_max_examples:
            break
            
        paper_ids = author_to_ids[author]
        if len(paper_ids) < 2:
            continue
        
        # Генерация примеров
        num_examples = min(max_examples_per_author, len(paper_ids))
        for _ in range(num_examples):
            if total_max_examples and total_examples >= total_max_examples:
                break
                
            current_id = random.choice(paper_ids)
            other_ids = [pid for pid in paper_ids if pid != current_id]
            if not other_ids:
                continue
                
            # Ограничиваем профиль
            sampled_others = random.sample(other_ids, min(max_profile_size, len(other_ids)))
            pos_id = random.choice(sampled_others)  # положительный — из профиля
            
            # Negative
            non_author_ids = list(all_ids - set(paper_ids))
            if not non_author_ids:
                continue
            neg_id = random.choice(non_author_ids)
            
            # Получаем данные
            cur_title = id_to_title.get(current_id, "")
            pos_title = id_to_title.get(pos_id, "")
            neg_title = id_to_title.get(neg_id, "")
            
            if not cur_title or not pos_title or not neg_title:
                continue
            
            # Случайный порядок
            if random.random() < 0.5:
                ref1, ref2 = neg_title, pos_title
                answer = "[2]"
            else:
                ref1, ref2 = pos_title, neg_title
                answer = "[1]"
            
            input_text = (
                f'For author {author} who has written the paper with the title "{cur_title}", '
                f'which reference is related? Just answer with [1] or [2] without explanation. '
                f'[1]: "{ref1}" [2]: "{ref2}"'
            )
            
            # behavior_profile: только sampled_others
            # behavior_texts = [
            #     f'TITLE: "{id_to_title.get(pid, "")}" ABSTRACT: {id_to_abstract.get(pid, "")}'
            #     for pid in sampled_others
            # ]
            behavior_texts = [
                f'TITLE: "{id_to_title.get(pid, "")}"'
                for pid in sampled_others
            ]

            results.append({
                "task": "LaMP_1",
                "id": str(current_id),
                "input_text": input_text,
                "output_text": answer,
                "behavior_profile_text": behavior_texts
            })
            total_examples += 1
    
    print(f"✅ Generated {len(results)} examples.")
    return results


if __name__ == "__main__":
    filepath = "/home/veltman.lina/.cache/kagglehub/datasets/nechbamohammed/research-papers-dataset/versions/1/dblp-v10.csv"
    print("📂 Loading data...")
    df = pd.read_csv(filepath)
    
    # Генерация (ограничим до 50K примеров, максимум 3 на автора)
    dataset = build_dataset_fast(
        df,
        max_examples_per_author=130,
        total_max_examples=1_000_000
    )
    
    # Сохранение
    output_path = "../data/kaggle_titles.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved to {output_path}")
