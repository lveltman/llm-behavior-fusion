import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from rouge_score import rouge_scorer

def compute_metric(metric_name, preds, labels, tokenizer):
    print(f"preds = {preds}")
    print(f"labels = {labels}")
    print()
    if metric_name == "accuracy":
        return accuracy_score(labels, preds)

    elif metric_name == "accuracy_f1":
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }

    elif metric_name == "regression":
        # preds_f = np.array([float(p.strip()) for p in preds])
        preds_f = np.array([float(p.strip()) if p.isdigit() else 0.0 for p in preds])
        labels_f = np.array([float(l.strip()) for l in labels])
        return {
            "mae": mean_absolute_error(labels_f, preds_f),
            "rmse": mean_squared_error(labels_f, preds_f)
        }

    elif metric_name == "rouge":
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = [scorer.score(r, p) for r, p in zip(labels, preds)]
        rouge1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
        rougeL = sum(s['rougeL'].fmeasure for s in scores) / len(scores)
        return {"ROUGE-1": rouge1, "ROUGE-L": rougeL}

    else:
        raise ValueError(f"Unknown metric: {metric_name}")
