import pandas as pd
import numpy as np
import re
import ast
import random
import json
from tqdm import tqdm
from collections import Counter

# -------------------------------------------------
# Парсеры авторов
# -------------------------------------------------

def parse_authors_new_format(authors_str):
    """Для формата: "['Name1', 'Name2']" → возвращает set"""
    if not isinstance(authors_str, str):
        return set()
    authors_str = authors_str.strip()
    if not (authors_str.startswith("[") and authors_str.endswith("]")):
        return set()
    try:
        lst = ast.literal_eval(authors_str)
        if isinstance(lst, list):
            return {str(x).strip() for x in lst if x and str(x).strip() != '...'}
    except:
        pass
    return set()

def parse_authors_as_list(authors_str):
    """Для формата: "['Name1', 'Name2']" → возвращает список с сохранением порядка"""
    if not isinstance(authors_str, str):
        return []
    authors_str = authors_str.strip()
    if authors_str.startswith("[") and authors_str.endswith("]"):
        try:
            lst = ast.literal_eval(authors_str)
            if isinstance(lst, list):
                return [str(x).strip() for x in lst if x and str(x).strip() != '...']
        except:
            pass
    return []


# -------------------------------------------------
# Генерация датасета
# -------------------------------------------------

def build_dataset(df, max_examples=1_000_000):
    """
    Генерирует LaMP-1 датасет.
    Режим: только 'new' (колонка 'authors').
    Берёт только первого автора и до 3 его статей в behavior_profile.
    """
    df = df.copy()
    
    if 'authors' not in df.columns:
        raise ValueError("Требуется колонка 'authors'")
    
    print("Парсинг авторов...")
    df['author_set'] = df['authors'].apply(parse_authors_new_format)
    df['author_list'] = df['authors'].apply(parse_authors_as_list)
    
    # Удаляем строки без авторов
    df = df[df['author_set'].apply(len) > 0].copy()
    df = df[df['author_list'].apply(len) > 0].copy()
    
    # Строим маппинг: автор → список индексов его статей
    author_to_papers = {}
    for idx, author_set in zip(df.index, df['author_set']):
        for author in author_set:
            author_to_papers.setdefault(author, []).append(idx)
    
    # Фильтрация: оставить только статьи, где хотя бы один автор имеет ≥2 статьи
    all_authors = [a for authors in df['author_set'] for a in authors]
    author_counts = Counter(all_authors)
    multi_authors = {a for a, cnt in author_counts.items() if cnt >= 2}
    df = df[df['author_set'].apply(lambda s: bool(s & multi_authors))].copy()
    print(f"Осталось статей после фильтрации: {len(df)}")

    # Предзагрузка данных в массивы (для скорости)
    titles = df['title'].fillna("").astype(str).values
    abstracts = df['abstract'].fillna("").astype(str).values
    ids = df['id'].astype(str).values
    author_sets = df['author_set'].values
    author_lists = df['author_list'].values
    
    indices = df.index.tolist()
    n = len(df)
    all_indices_set = set(indices)
    index_to_pos = {idx: i for i, idx in enumerate(indices)}
    
    # Предвычислим маппинг для positive/negative (все статьи соавторов текущей статьи)
    author_papers_list = []
    for author_set in df['author_set']:
        papers = set()
        for author in author_set:
            papers.update(author_to_papers.get(author, []))
        author_papers_list.append(papers)
    
    results = []
    print("Генерация датасета (только первый автор, до 3 behavior-статей)...")
    
    for i in tqdm(range(n), desc="Обработка"):
        idx = indices[i]
        title = titles[i]
        if not title.strip():
            continue

        author_papers = author_papers_list[i]
        other_author_papers = author_papers - {idx}
        if not other_author_papers:
            continue

        # Positive: случайная статья от того же автора
        pos_paper_idx = random.choice(list(other_author_papers))
        pos_pos = index_to_pos[pos_paper_idx]
        pos_title = titles[pos_pos]
        if not pos_title.strip():
            continue

        # Negative: случайная статья от другого автора
        non_author_papers = list(all_indices_set - author_papers)
        if not non_author_papers:
            continue
        neg_paper_idx = random.choice(non_author_papers)
        neg_pos = index_to_pos[neg_paper_idx]
        neg_title = titles[neg_pos]
        if not neg_title.strip():
            continue

        # Случайный порядок референсов
        if random.random() < 0.5:
            ref1, ref2 = neg_title, pos_title
            answer = "[2]"
        else:
            ref1, ref2 = pos_title, neg_title
            answer = "[1]"

        # Только первый автор
        first_author = author_lists[i][0]
        input_text = (
            f'For author {first_author} who has written the paper with the title "{title}", '
            f'which reference is related? Just answer with [1] or [2] without explanation. '
            f'[1]: "{ref1}" [2]: "{ref2}"'
        )

        # Behavior profile: до 3 статей первого автора (кроме текущей)
        behavior_texts = []
        author_papers_all = author_to_papers.get(first_author, [])
        relevant_papers = [pid for pid in author_papers_all if pid != idx]
        k = min(3, len(relevant_papers))
        if k > 0:
            sampled_papers = random.sample(relevant_papers, k)
            for paper_idx in sampled_papers:
                p_pos = index_to_pos[paper_idx]
                t = titles[p_pos]
                a = abstracts[p_pos]
                if t.strip():
                    behavior_texts.append(f'TITLE: "{t}" ABSTRACT: {a}')

        results.append({
            "task": "LaMP_1",
            "id": ids[i],
            "input_text": input_text,
            "output_text": answer,
            "behavior_profile_text": behavior_texts
        })

        if len(results) >= max_examples:
            break

    print(f"✅ Сгенерировано {len(results)} примеров.")
    return results



if __name__ == "__main__":
    filepath = "/home/veltman.lina/.cache/kagglehub/datasets/nechbamohammed/research-papers-dataset/versions/1/dblp-v10.csv"
    
    print("Загрузка данных...")
    df = pd.read_csv(filepath, usecols=['id', 'title', 'abstract', 'authors'])
    
    dataset = build_dataset(df, max_examples=500_000)
    
    output_path = "../data/lamp1_author_relevance_first_author.json"
    print(f"Сохранение в {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print("Готово! 🎉")