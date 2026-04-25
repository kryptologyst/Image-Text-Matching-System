"""Visualization utilities for image-text matching."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union

from ..utils.device import get_device


class ImageTextVisualizer:
    """
    Visualization utilities for image-text matching results.
    
    Provides methods for visualizing attention maps, similarity matrices,
    retrieval results, and training progress.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        processor: object,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize visualizer.
        
        Args:
            model: Trained model for visualization
            processor: CLIP processor for preprocessing
            device: Device for computation
        """
        self.model = model
        self.processor = processor
        self.device = device or get_device("auto")
        self.model.to(self.device)
        self.model.eval()
    
    def visualize_attention_maps(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        text: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """
        Visualize attention maps for image-text interaction.
        
        Args:
            image: Input image (path, PIL Image, or tensor)
            text: Input text
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Preprocess inputs
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            image = self._tensor_to_pil(image)
        
        inputs = self.processor(text=[text], images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            attention_weights = self.model.get_attention_weights(
                pixel_values, input_ids, attention_mask
            )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Vision attention (average across layers)
        vision_attentions = attention_weights["vision_attentions"]
        avg_vision_attention = torch.mean(
            torch.stack([attn[0] for attn in vision_attentions]), dim=0
        )
        
        # Reshape attention to spatial dimensions
        patch_size = int(np.sqrt(avg_vision_attention.shape[-1]))
        attention_map = avg_vision_attention[0, 1:].reshape(patch_size, patch_size)
        
        im1 = axes[1].imshow(attention_map.cpu().numpy(), cmap="hot", interpolation="nearest")
        axes[1].set_title("Vision Attention")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1])
        
        # Text attention (average across layers)
        text_attentions = attention_weights["text_attentions"]
        avg_text_attention = torch.mean(
            torch.stack([attn[0] for attn in text_attentions]), dim=0
        )
        
        # Average attention across all heads
        text_attention_map = torch.mean(avg_text_attention, dim=0)
        
        # Create text tokens for x-axis
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        im2 = axes[2].imshow(text_attention_map.cpu().numpy(), cmap="hot", aspect="auto")
        axes[2].set_title("Text Attention")
        axes[2].set_xlabel("Token Position")
        axes[2].set_ylabel("Token Position")
        axes[2].set_xticks(range(len(tokens)))
        axes[2].set_xticklabels(tokens, rotation=45, ha="right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def visualize_similarity_matrix(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        """
        Visualize similarity matrix between images and texts.
        
        Args:
            images: List of input images
            texts: List of input texts
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Preprocess inputs
        processed_images = []
        processed_texts = []
        
        for image in images:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, torch.Tensor):
                image = self._tensor_to_pil(image)
            processed_images.append(image)
        
        # Get embeddings
        image_embeds = []
        text_embeds = []
        
        with torch.no_grad():
            for image in processed_images:
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                embed = self.model.encode_image(pixel_values)
                image_embeds.append(embed.cpu())
            
            for text in texts:
                inputs = self.processor(text=[text], return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                embed = self.model.encode_text(input_ids, attention_mask)
                text_embeds.append(embed.cpu())
        
        # Compute similarity matrix
        image_embeds = torch.cat(image_embeds, dim=0)
        text_embeds = torch.cat(text_embeds, dim=0)
        
        similarity_matrix = torch.matmul(image_embeds, text_embeds.t()).numpy()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(similarity_matrix, cmap="viridis", aspect="auto")
        
        # Set labels
        ax.set_xlabel("Texts")
        ax.set_ylabel("Images")
        ax.set_title("Image-Text Similarity Matrix")
        
        # Set tick labels
        ax.set_xticks(range(len(texts)))
        ax.set_xticklabels([f"T{i+1}" for i in range(len(texts))], rotation=45)
        ax.set_yticks(range(len(images)))
        ax.set_yticklabels([f"I{i+1}" for i in range(len(images))])
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add text annotations
        for i in range(len(images)):
            for j in range(len(texts)):
                text = ax.text(
                    j, i, f"{similarity_matrix[i, j]:.2f}",
                    ha="center", va="center", color="white"
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def visualize_retrieval_results(
        self,
        query_image: Union[str, Image.Image],
        candidate_texts: List[str],
        top_k: int = 5,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> plt.Figure:
        """
        Visualize retrieval results for a query image.
        
        Args:
            query_image: Query image
            candidate_texts: List of candidate texts
            top_k: Number of top results to show
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Preprocess query image
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert("RGB")
        elif isinstance(query_image, torch.Tensor):
            query_image = self._tensor_to_pil(query_image)
        
        # Get embeddings
        with torch.no_grad():
            # Query image embedding
            inputs = self.processor(images=query_image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            query_embed = self.model.encode_image(pixel_values)
            
            # Candidate text embeddings
            candidate_embeds = []
            for text in candidate_texts:
                inputs = self.processor(text=[text], return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                embed = self.model.encode_text(input_ids, attention_mask)
                candidate_embeds.append(embed.cpu())
        
        # Compute similarities
        candidate_embeds = torch.cat(candidate_embeds, dim=0)
        similarities = torch.matmul(query_embed, candidate_embeds.t()).squeeze()
        
        # Get top-k results
        top_k_indices = torch.topk(similarities, min(top_k, len(candidate_texts))).indices
        
        # Create visualization
        fig, axes = plt.subplots(2, (top_k + 1) // 2, figsize=figsize)
        if (top_k + 1) // 2 == 1:
            axes = axes.reshape(-1, 1)
        
        # Show query image
        axes[0, 0].imshow(query_image)
        axes[0, 0].set_title("Query Image", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")
        
        # Show top-k results
        for i, idx in enumerate(top_k_indices):
            row = (i + 1) // ((top_k + 1) // 2)
            col = (i + 1) % ((top_k + 1) // 2)
            
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].text(
                    0.5, 0.5, candidate_texts[idx],
                    ha="center", va="center",
                    fontsize=10, wrap=True,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue")
                )
                axes[row, col].set_title(
                    f"Rank {i+1}: {similarities[idx]:.3f}",
                    fontsize=10
                )
                axes[row, col].axis("off")
        
        plt.suptitle("Image-to-Text Retrieval Results", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def visualize_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        metrics: Optional[Dict[str, List[float]]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """
        Visualize training curves.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses (optional)
            metrics: Validation metrics (optional)
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss curves
        epochs = range(1, len(train_losses) + 1)
        axes[0].plot(epochs, train_losses, label="Train Loss", marker="o")
        
        if val_losses:
            axes[0].plot(epochs, val_losses, label="Val Loss", marker="s")
        
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        # Metrics curves
        if metrics:
            for metric_name, metric_values in metrics.items():
                axes[1].plot(epochs, metric_values, label=metric_name, marker="o")
            
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Metric Value")
            axes[1].set_title("Validation Metrics")
            axes[1].legend()
            axes[1].grid(True)
        else:
            axes[1].text(0.5, 0.5, "No metrics available", ha="center", va="center")
            axes[1].set_title("Validation Metrics")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def create_retrieval_gallery(
        self,
        query_text: str,
        candidate_images: List[Union[str, Image.Image]],
        top_k: int = 5,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
    ) -> plt.Figure:
        """
        Create a gallery of retrieved images for a text query.
        
        Args:
            query_text: Query text
            candidate_images: List of candidate images
            top_k: Number of top results to show
            save_path: Path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Preprocess candidate images
        processed_images = []
        for image in candidate_images:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, torch.Tensor):
                image = self._tensor_to_pil(image)
            processed_images.append(image)
        
        # Get embeddings
        with torch.no_grad():
            # Query text embedding
            inputs = self.processor(text=[query_text], return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            query_embed = self.model.encode_text(input_ids, attention_mask)
            
            # Candidate image embeddings
            candidate_embeds = []
            for image in processed_images:
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                embed = self.model.encode_image(pixel_values)
                candidate_embeds.append(embed.cpu())
        
        # Compute similarities
        candidate_embeds = torch.cat(candidate_embeds, dim=0)
        similarities = torch.matmul(query_embed, candidate_embeds.t()).squeeze()
        
        # Get top-k results
        top_k_indices = torch.topk(similarities, min(top_k, len(candidate_images))).indices
        
        # Create visualization
        cols = min(5, top_k)
        rows = (top_k + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Show query text
        axes[0, 0].text(
            0.5, 0.5, f"Query: {query_text}",
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"),
            wrap=True
        )
        axes[0, 0].set_title("Query Text", fontsize=12, fontweight="bold")
        axes[0, 0].axis("off")
        
        # Show top-k results
        for i, idx in enumerate(top_k_indices):
            if i == 0:
                continue  # Skip first position (query)
            
            row = i // cols
            col = i % cols
            
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].imshow(processed_images[idx])
                axes[row, col].set_title(
                    f"Rank {i}: {similarities[idx]:.3f}",
                    fontsize=10
                )
                axes[row, col].axis("off")
        
        # Hide unused subplots
        for i in range(top_k, rows * cols):
            row = i // cols
            col = i % cols
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis("off")
        
        plt.suptitle("Text-to-Image Retrieval Results", fontsize=16, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        return fig
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        # Denormalize if needed
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2
        
        # Convert to numpy
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
        tensor = np.clip(tensor, 0, 1)
        
        # Convert to PIL
        return Image.fromarray((tensor * 255).astype(np.uint8))
