# metrics.py

import numpy as np
import math

def accuracy_at_k(probs: np.ndarray, labels: np.ndarray, k: int):
    """
    计算 Accuracy@K
    """
    topk_indices = np.argsort(-probs, axis=1)[:, :k]
    hits = 0
    for i in range(len(labels)):
        if labels[i] in topk_indices[i]:
            hits += 1
    return hits / len(labels)

def precision_at_k(probs: np.ndarray, labels: np.ndarray, k: int):
    """
    计算 Precision@K
    在单标签分类中，Precision@K 与 Accuracy@K 相同
    """
    return accuracy_at_k(probs, labels, k)

def recall_at_k(probs: np.ndarray, labels: np.ndarray, k: int):
    """
    计算 Recall@K
    在单标签分类中，Recall@K 与 Accuracy@K 相同
    """
    return accuracy_at_k(probs, labels, k)

def f1_score_at_k(precision, recall):
    """
    计算 F1 Score
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def map_at_k(probs: np.ndarray, labels: np.ndarray, k: int):
    """
    计算 MAP@K
    简化实现：对于每个样本，如果标签在 Top-K 中，则计算 1 / rank
    """
    map_sum = 0.0
    for i in range(len(labels)):
        topk = np.argsort(-probs[i])[:k]
        if labels[i] in topk:
            rank = np.where(topk == labels[i])[0][0] + 1  # 1-based rank
            map_sum += 1.0 / rank
    return map_sum / len(labels)

def ndcg_at_k(probs: np.ndarray, labels: np.ndarray, k: int):
    """
    计算 NDCG@K
    简化实现：对于每个样本，如果标签在 Top-K 中，则根据其 rank 计算 DCG
    """
    dcg_sum = 0.0
    for i in range(len(labels)):
        topk = np.argsort(-probs[i])[:k]
        if labels[i] in topk:
            rank = np.where(topk == labels[i])[0][0] + 1  # 1-based rank
            dcg_sum += 1.0 / math.log2(rank + 1)
    # IDCG: 理想情况下，每个样本的标签在第一位
    idcg = 1.0  # 因为每个样本只有一个标签
    return dcg_sum / len(labels)

def mse(probs: np.ndarray, labels: np.ndarray):
    """
    计算均方误差
    """
    preds = np.argmax(probs, axis=1)
    return np.mean((preds - labels) ** 2)
