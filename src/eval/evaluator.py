"""Evaluation metrics and utilities for image-text matching."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .metrics import compute_retrieval_metrics


class ImageTextEvaluator:
    """
    Evaluator for image-text matching tasks.
    
    Computes various retrieval metrics and provides comprehensive evaluation.
    """
    
    def __init__(
        self,
        metrics: List[str] = None,
        evaluate_image_to_text: bool = True,
        evaluate_text_to_image: bool = True,
        evaluate_bidirectional: bool = True,
        batch_size: int = 64,
        num_workers: int = 4,
        top_k_values: List[int] = None,
        similarity_metric: str = "cosine",
        normalize_embeddings: bool = True,
        save_predictions: bool = True,
        save_similarity_matrix: bool = False,
        save_attention_maps: bool = False,
        output_dir: str = "evaluation_results",
        create_retrieval_plots: bool = True,
        create_confusion_matrix: bool = False,
        max_examples_to_visualize: int = 100,
    ):
        """
        Initialize evaluator.
        
        Args:
            metrics: List of metrics to compute
            evaluate_image_to_text: Whether to evaluate image-to-text retrieval
            evaluate_text_to_image: Whether to evaluate text-to-image retrieval
            evaluate_bidirectional: Whether to evaluate bidirectional retrieval
            batch_size: Batch size for evaluation
            num_workers: Number of workers for data loading
            top_k_values: Top-k values for recall computation
            similarity_metric: Similarity metric for retrieval
            normalize_embeddings: Whether to normalize embeddings
            save_predictions: Whether to save predictions
            save_similarity_matrix: Whether to save similarity matrix
            save_attention_maps: Whether to save attention maps
            output_dir: Output directory for results
            create_retrieval_plots: Whether to create retrieval plots
            create_confusion_matrix: Whether to create confusion matrix
            max_examples_to_visualize: Maximum examples to visualize
        """
        self.metrics = metrics or [
            "recall_at_1",
            "recall_at_5", 
            "recall_at_10",
            "median_rank",
            "mean_rank",
            "mean_average_precision",
        ]
        self.evaluate_image_to_text = evaluate_image_to_text
        self.evaluate_text_to_image = evaluate_text_to_image
        self.evaluate_bidirectional = evaluate_bidirectional
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.top_k_values = top_k_values or [1, 5, 10, 50, 100]
        self.similarity_metric = similarity_metric
        self.normalize_embeddings = normalize_embeddings
        self.save_predictions = save_predictions
        self.save_similarity_matrix = save_similarity_matrix
        self.save_attention_maps = save_attention_maps
        self.output_dir = output_dir
        self.create_retrieval_plots = create_retrieval_plots
        self.create_confusion_matrix = create_confusion_matrix
        self.max_examples_to_visualize = max_examples_to_visualize
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluate model on dataset.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            device: Device for computation
            
        Returns:
            Dictionary containing evaluation results
        """
        model.eval()
        
        # Collect all embeddings and metadata
        all_image_embeds = []
        all_text_embeds = []
        all_image_ids = []
        all_texts = []
        all_image_paths = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                pixel_values = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Get embeddings
                outputs = model(pixel_values, input_ids, attention_mask)
                image_embeds = outputs["image_embeds"]
                text_embeds = outputs["text_embeds"]
                
                # Normalize embeddings if specified
                if self.normalize_embeddings:
                    image_embeds = F.normalize(image_embeds, p=2, dim=-1)
                    text_embeds = F.normalize(text_embeds, p=2, dim=-1)
                
                # Store embeddings and metadata
                all_image_embeds.append(image_embeds.cpu())
                all_text_embeds.append(text_embeds.cpu())
                all_image_ids.extend(batch["image_id"])
                all_texts.extend(batch["text"])
                all_image_paths.extend(batch["image_path"])
        
        # Concatenate all embeddings
        all_image_embeds = torch.cat(all_image_embeds, dim=0)
        all_text_embeds = torch.cat(all_text_embeds, dim=0)
        
        # Compute evaluation results
        results = {}
        
        if self.evaluate_image_to_text:
            i2t_results = self._evaluate_image_to_text(
                all_image_embeds, all_text_embeds
            )
            results["image_to_text"] = i2t_results
        
        if self.evaluate_text_to_image:
            t2i_results = self._evaluate_text_to_image(
                all_image_embeds, all_text_embeds
            )
            results["text_to_image"] = t2i_results
        
        if self.evaluate_bidirectional:
            bidirectional_results = self._evaluate_bidirectional(
                all_image_embeds, all_text_embeds
            )
            results["bidirectional"] = bidirectional_results
        
        # Save results if requested
        if self.save_predictions:
            self._save_predictions(
                results,
                all_image_ids,
                all_texts,
                all_image_paths,
            )
        
        # Create visualizations if requested
        if self.create_retrieval_plots:
            self._create_retrieval_plots(results)
        
        return results
    
    def _evaluate_image_to_text(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate image-to-text retrieval."""
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(
            image_embeds, text_embeds
        )
        
        # Compute metrics
        metrics = compute_retrieval_metrics(
            similarity_matrix,
            self.top_k_values,
            self.metrics,
        )
        
        return metrics
    
    def _evaluate_text_to_image(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate text-to-image retrieval."""
        # Compute similarity matrix (transpose for text-to-image)
        similarity_matrix = self._compute_similarity_matrix(
            text_embeds, image_embeds
        )
        
        # Compute metrics
        metrics = compute_retrieval_metrics(
            similarity_matrix,
            self.top_k_values,
            self.metrics,
        )
        
        return metrics
    
    def _evaluate_bidirectional(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Dict[str, float]:
        """Evaluate bidirectional retrieval."""
        # Compute both directions
        i2t_results = self._evaluate_image_to_text(image_embeds, text_embeds)
        t2i_results = self._evaluate_text_to_image(image_embeds, text_embeds)
        
        # Average metrics
        bidirectional_results = {}
        for metric in i2t_results:
            bidirectional_results[metric] = (
                i2t_results[metric] + t2i_results[metric]
            ) / 2
        
        return bidirectional_results
    
    def _compute_similarity_matrix(
        self,
        query_embeds: torch.Tensor,
        candidate_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity matrix between query and candidate embeddings."""
        if self.similarity_metric == "cosine":
            similarity_matrix = torch.matmul(query_embeds, candidate_embeds.t())
        elif self.similarity_metric == "dot_product":
            similarity_matrix = torch.matmul(query_embeds, candidate_embeds.t())
        elif self.similarity_metric == "euclidean":
            # Convert to distance (negative for higher is better)
            distances = torch.cdist(query_embeds, candidate_embeds, p=2)
            similarity_matrix = -distances
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarity_matrix
    
    def _save_predictions(
        self,
        results: Dict[str, Dict[str, float]],
        image_ids: List[str],
        texts: List[str],
        image_paths: List[str],
    ) -> None:
        """Save prediction results."""
        import json
        from pathlib import Path
        
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_path / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save metadata
        metadata = {
            "image_ids": image_ids,
            "texts": texts,
            "image_paths": image_paths,
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _create_retrieval_plots(self, results: Dict[str, Dict[str, float]]) -> None:
        """Create retrieval visualization plots."""
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create recall curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot recall at different k values
        k_values = self.top_k_values
        for direction, metrics in results.items():
            if direction == "bidirectional":
                continue
            
            recall_values = []
            for k in k_values:
                metric_name = f"recall_at_{k}"
                if metric_name in metrics:
                    recall_values.append(metrics[metric_name])
                else:
                    recall_values.append(0.0)
            
            axes[0].plot(k_values, recall_values, marker="o", label=direction)
        
        axes[0].set_xlabel("Top-k")
        axes[0].set_ylabel("Recall")
        axes[0].set_title("Recall@k Curves")
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot median rank comparison
        directions = list(results.keys())
        median_ranks = [results[d].get("median_rank", 0) for d in directions]
        
        axes[1].bar(directions, median_ranks)
        axes[1].set_ylabel("Median Rank")
        axes[1].set_title("Median Rank Comparison")
        axes[1].grid(True, axis="y")
        
        plt.tight_layout()
        plt.savefig(output_path / "retrieval_plots.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def create_leaderboard(
        self,
        results: Dict[str, Dict[str, float]],
        model_name: str = "Model",
    ) -> str:
        """
        Create a formatted leaderboard from results.
        
        Args:
            results: Evaluation results
            model_name: Name of the model
            
        Returns:
            Formatted leaderboard string
        """
        leaderboard = f"\n{'='*60}\n"
        leaderboard += f"Image-Text Matching Leaderboard\n"
        leaderboard += f"{'='*60}\n"
        leaderboard += f"Model: {model_name}\n\n"
        
        for direction, metrics in results.items():
            leaderboard += f"{direction.replace('_', ' ').title()}:\n"
            leaderboard += f"{'-'*40}\n"
            
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    leaderboard += f"{metric_name:20}: {metric_value:.4f}\n"
                else:
                    leaderboard += f"{metric_name:20}: {metric_value}\n"
            
            leaderboard += "\n"
        
        leaderboard += f"{'='*60}\n"
        
        return leaderboard
