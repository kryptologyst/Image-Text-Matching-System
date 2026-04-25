"""Retrieval metrics computation for image-text matching."""

import numpy as np
import torch
from typing import Dict, List, Union


def compute_retrieval_metrics(
    similarity_matrix: torch.Tensor,
    top_k_values: List[int],
    metrics: List[str],
) -> Dict[str, float]:
    """
    Compute retrieval metrics from similarity matrix.
    
    Args:
        similarity_matrix: Similarity scores between queries and candidates
        top_k_values: List of k values for recall@k computation
        metrics: List of metrics to compute
        
    Returns:
        Dictionary containing computed metrics
    """
    results = {}
    
    # Convert to numpy for easier computation
    similarities = similarity_matrix.cpu().numpy()
    batch_size = similarities.shape[0]
    
    # Create ground truth labels (diagonal elements are correct matches)
    ground_truth = np.arange(batch_size)
    
    # Sort similarities in descending order
    sorted_indices = np.argsort(-similarities, axis=1)
    
    # Compute recall@k for each k value
    for k in top_k_values:
        if f"recall_at_{k}" in metrics:
            recall_at_k = compute_recall_at_k(sorted_indices, ground_truth, k)
            results[f"recall_at_{k}"] = recall_at_k
    
    # Compute median rank
    if "median_rank" in metrics:
        median_rank = compute_median_rank(sorted_indices, ground_truth)
        results["median_rank"] = median_rank
    
    # Compute mean rank
    if "mean_rank" in metrics:
        mean_rank = compute_mean_rank(sorted_indices, ground_truth)
        results["mean_rank"] = mean_rank
    
    # Compute mean average precision
    if "mean_average_precision" in metrics:
        map_score = compute_mean_average_precision(sorted_indices, ground_truth)
        results["mean_average_precision"] = map_score
    
    return results


def compute_recall_at_k(
    sorted_indices: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """
    Compute recall@k metric.
    
    Args:
        sorted_indices: Indices sorted by similarity (descending)
        ground_truth: Ground truth indices
        k: Number of top results to consider
        
    Returns:
        Recall@k score
    """
    # Check if ground truth is in top-k results
    top_k_indices = sorted_indices[:, :k]
    hits = np.any(top_k_indices == ground_truth[:, np.newaxis], axis=1)
    
    return np.mean(hits).item()


def compute_median_rank(
    sorted_indices: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute median rank of ground truth results.
    
    Args:
        sorted_indices: Indices sorted by similarity (descending)
        ground_truth: Ground truth indices
        
    Returns:
        Median rank
    """
    ranks = []
    for i, gt_idx in enumerate(ground_truth):
        rank = np.where(sorted_indices[i] == gt_idx)[0][0] + 1  # 1-indexed
        ranks.append(rank)
    
    return np.median(ranks).item()


def compute_mean_rank(
    sorted_indices: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute mean rank of ground truth results.
    
    Args:
        sorted_indices: Indices sorted by similarity (descending)
        ground_truth: Ground truth indices
        
    Returns:
        Mean rank
    """
    ranks = []
    for i, gt_idx in enumerate(ground_truth):
        rank = np.where(sorted_indices[i] == gt_idx)[0][0] + 1  # 1-indexed
        ranks.append(rank)
    
    return np.mean(ranks).item()


def compute_mean_average_precision(
    sorted_indices: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute mean average precision (mAP).
    
    Args:
        sorted_indices: Indices sorted by similarity (descending)
        ground_truth: Ground truth indices
        
    Returns:
        Mean average precision score
    """
    average_precisions = []
    
    for i, gt_idx in enumerate(ground_truth):
        # Find rank of ground truth
        gt_rank = np.where(sorted_indices[i] == gt_idx)[0][0] + 1  # 1-indexed
        
        # Compute average precision
        # AP = 1 / rank (since there's only one relevant item)
        ap = 1.0 / gt_rank
        average_precisions.append(ap)
    
    return np.mean(average_precisions).item()


def compute_precision_at_k(
    sorted_indices: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """
    Compute precision@k metric.
    
    Args:
        sorted_indices: Indices sorted by similarity (descending)
        ground_truth: Ground truth indices
        k: Number of top results to consider
        
    Returns:
        Precision@k score
    """
    # Check if ground truth is in top-k results
    top_k_indices = sorted_indices[:, :k]
    hits = np.any(top_k_indices == ground_truth[:, np.newaxis], axis=1)
    
    # Precision = hits / k
    precision = np.mean(hits) / k
    
    return precision.item()


def compute_ndcg_at_k(
    sorted_indices: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """
    Compute normalized discounted cumulative gain@k (NDCG@k).
    
    Args:
        sorted_indices: Indices sorted by similarity (descending)
        ground_truth: Ground truth indices
        k: Number of top results to consider
        
    Returns:
        NDCG@k score
    """
    ndcg_scores = []
    
    for i, gt_idx in enumerate(ground_truth):
        # Find rank of ground truth
        gt_rank = np.where(sorted_indices[i] == gt_idx)[0][0] + 1  # 1-indexed
        
        if gt_rank <= k:
            # DCG = 1 / log2(rank + 1)
            dcg = 1.0 / np.log2(gt_rank + 1)
            
            # IDCG = 1 / log2(2) = 1 (perfect ranking)
            idcg = 1.0
            
            # NDCG = DCG / IDCG
            ndcg = dcg / idcg
        else:
            ndcg = 0.0
        
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores).item()


def compute_hit_rate_at_k(
    sorted_indices: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """
    Compute hit rate@k metric (same as recall@k for single relevant item).
    
    Args:
        sorted_indices: Indices sorted by similarity (descending)
        ground_truth: Ground truth indices
        k: Number of top results to consider
        
    Returns:
        Hit rate@k score
    """
    return compute_recall_at_k(sorted_indices, ground_truth, k)


def compute_reciprocal_rank(
    sorted_indices: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute mean reciprocal rank (MRR).
    
    Args:
        sorted_indices: Indices sorted by similarity (descending)
        ground_truth: Ground truth indices
        
    Returns:
        Mean reciprocal rank
    """
    reciprocal_ranks = []
    
    for i, gt_idx in enumerate(ground_truth):
        # Find rank of ground truth
        gt_rank = np.where(sorted_indices[i] == gt_idx)[0][0] + 1  # 1-indexed
        
        # Reciprocal rank = 1 / rank
        rr = 1.0 / gt_rank
        reciprocal_ranks.append(rr)
    
    return np.mean(reciprocal_ranks).item()
