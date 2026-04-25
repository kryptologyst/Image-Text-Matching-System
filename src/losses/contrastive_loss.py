"""Loss functions for image-text matching."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for image-text matching.
    
    Implements InfoNCE loss with temperature scaling and optional
    hard negative mining.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        hard_negative_mining: bool = True,
        hard_negative_ratio: float = 0.5,
        margin: float = 0.2,
    ):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature for scaling logits
            hard_negative_mining: Whether to use hard negative mining
            hard_negative_ratio: Ratio of hard negatives to use
            margin: Margin for hard negative mining
        """
        super().__init__()
        self.temperature = temperature
        self.hard_negative_mining = hard_negative_mining
        self.hard_negative_ratio = hard_negative_ratio
        self.margin = margin
    
    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            logits_per_image: Image-to-text similarity logits
            logits_per_text: Text-to-image similarity logits
            
        Returns:
            Contrastive loss value
        """
        batch_size = logits_per_image.shape[0]
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # Compute losses
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        # Apply hard negative mining if enabled
        if self.hard_negative_mining:
            loss_i2t = self._apply_hard_negative_mining(
                logits_per_image, labels, loss_i2t
            )
            loss_t2i = self._apply_hard_negative_mining(
                logits_per_text, labels, loss_t2i
            )
        
        # Average the two losses
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss
    
    def _apply_hard_negative_mining(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply hard negative mining to focus on difficult examples.
        
        Args:
            logits: Similarity logits
            labels: Ground truth labels
            base_loss: Base cross-entropy loss
            
        Returns:
            Modified loss with hard negative mining
        """
        # Get probabilities
        probs = F.softmax(logits / self.temperature, dim=1)
        
        # Find hard negatives (high probability for wrong classes)
        hard_negatives = []
        for i in range(logits.shape[0]):
            # Get probabilities for wrong classes
            wrong_probs = probs[i]
            wrong_probs[labels[i]] = 0  # Remove positive class
            
            # Find hardest negatives
            num_hard = int(self.hard_negative_ratio * (logits.shape[1] - 1))
            if num_hard > 0:
                hard_indices = torch.topk(wrong_probs, num_hard).indices
                hard_negatives.append(hard_indices)
        
        # Compute weighted loss focusing on hard negatives
        if hard_negatives:
            weights = torch.ones_like(logits)
            for i, hard_indices in enumerate(hard_negatives):
                weights[i, hard_indices] *= 2.0  # Double weight for hard negatives
            
            # Apply weights to logits
            weighted_logits = logits * weights
            loss = F.cross_entropy(weighted_logits, labels)
        else:
            loss = base_loss
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet loss for image-text matching.
    
    Uses anchor-positive-negative triplets to learn embeddings.
    """
    
    def __init__(
        self,
        margin: float = 0.2,
        distance_metric: str = "cosine",
        mining_strategy: str = "hard",
    ):
        """
        Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            distance_metric: Distance metric ('cosine', 'euclidean')
            mining_strategy: Mining strategy ('hard', 'semi-hard', 'all')
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.mining_strategy = mining_strategy
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            image_embeds: Image embeddings
            text_embeds: Text embeddings
            
        Returns:
            Triplet loss value
        """
        # Compute pairwise distances
        distances = self._compute_distances(image_embeds, text_embeds)
        
        # Create positive and negative masks
        batch_size = image_embeds.shape[0]
        positive_mask = torch.eye(batch_size, device=image_embeds.device).bool()
        negative_mask = ~positive_mask
        
        # Extract positive and negative distances
        positive_distances = distances[positive_mask]
        negative_distances = distances[negative_mask].view(batch_size, batch_size - 1)
        
        # Apply mining strategy
        if self.mining_strategy == "hard":
            # Use hardest negatives
            hardest_negative_distances, _ = torch.max(negative_distances, dim=1)
            loss = F.relu(positive_distances - hardest_negative_distances + self.margin)
        elif self.mining_strategy == "semi-hard":
            # Use semi-hard negatives
            semi_hard_mask = negative_distances > positive_distances.unsqueeze(1)
            semi_hard_mask = semi_hard_mask & (negative_distances < positive_distances.unsqueeze(1) + self.margin)
            
            if semi_hard_mask.any():
                semi_hard_distances = negative_distances[semi_hard_mask]
                loss = F.relu(positive_distances.unsqueeze(1) - semi_hard_distances + self.margin)
            else:
                loss = torch.tensor(0.0, device=image_embeds.device)
        else:  # all
            # Use all negatives
            loss = F.relu(
                positive_distances.unsqueeze(1) - negative_distances + self.margin
            )
        
        return loss.mean()
    
    def _compute_distances(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise distances between embeddings.
        
        Args:
            image_embeds: Image embeddings
            text_embeds: Text embeddings
            
        Returns:
            Pairwise distance matrix
        """
        if self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            similarities = F.cosine_similarity(
                image_embeds.unsqueeze(1), text_embeds.unsqueeze(0), dim=2
            )
            distances = 1 - similarities
        elif self.distance_metric == "euclidean":
            # Euclidean distance
            distances = torch.cdist(image_embeds, text_embeds, p=2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances


class MarginLoss(nn.Module):
    """
    Margin loss for image-text matching.
    
    Uses margin-based ranking loss to separate positive and negative pairs.
    """
    
    def __init__(
        self,
        margin: float = 0.2,
        similarity_metric: str = "cosine",
    ):
        """
        Initialize margin loss.
        
        Args:
            margin: Margin for ranking loss
            similarity_metric: Similarity metric ('cosine', 'dot_product')
        """
        super().__init__()
        self.margin = margin
        self.similarity_metric = similarity_metric
    
    def forward(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute margin loss.
        
        Args:
            image_embeds: Image embeddings
            text_embeds: Text embeddings
            
        Returns:
            Margin loss value
        """
        batch_size = image_embeds.shape[0]
        
        # Compute similarities
        similarities = self._compute_similarities(image_embeds, text_embeds)
        
        # Create positive and negative masks
        positive_mask = torch.eye(batch_size, device=image_embeds.device).bool()
        negative_mask = ~positive_mask
        
        # Extract positive and negative similarities
        positive_similarities = similarities[positive_mask]
        negative_similarities = similarities[negative_mask].view(batch_size, batch_size - 1)
        
        # Compute margin loss
        # Loss = max(0, margin - positive_similarity + negative_similarity)
        loss = F.relu(
            self.margin - positive_similarities.unsqueeze(1) + negative_similarities
        )
        
        return loss.mean()
    
    def _compute_similarities(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise similarities between embeddings.
        
        Args:
            image_embeds: Image embeddings
            text_embeds: Text embeddings
            
        Returns:
            Pairwise similarity matrix
        """
        if self.similarity_metric == "cosine":
            similarities = F.cosine_similarity(
                image_embeds.unsqueeze(1), text_embeds.unsqueeze(0), dim=2
            )
        elif self.similarity_metric == "dot_product":
            similarities = torch.matmul(image_embeds, text_embeds.t())
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        return similarities


class CombinedLoss(nn.Module):
    """
    Combined loss function using multiple loss components.
    
    Combines contrastive loss with regularization terms.
    """
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        regularization_weight: float = 0.01,
        temperature: float = 0.07,
        **kwargs,
    ):
        """
        Initialize combined loss.
        
        Args:
            contrastive_weight: Weight for contrastive loss
            regularization_weight: Weight for regularization
            temperature: Temperature for contrastive loss
            **kwargs: Additional arguments for loss functions
        """
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.regularization_weight = regularization_weight
        
        self.contrastive_loss = ContrastiveLoss(temperature=temperature, **kwargs)
    
    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            logits_per_image: Image-to-text similarity logits
            logits_per_text: Text-to-image similarity logits
            image_embeds: Image embeddings
            text_embeds: Text embeddings
            
        Returns:
            Dictionary containing loss components
        """
        # Contrastive loss
        contrastive_loss = self.contrastive_loss(logits_per_image, logits_per_text)
        
        # Regularization loss (L2 penalty on embeddings)
        image_reg = torch.mean(torch.norm(image_embeds, p=2, dim=1))
        text_reg = torch.mean(torch.norm(text_embeds, p=2, dim=1))
        regularization_loss = image_reg + text_reg
        
        # Combined loss
        total_loss = (
            self.contrastive_weight * contrastive_loss +
            self.regularization_weight * regularization_loss
        )
        
        return {
            "total_loss": total_loss,
            "contrastive_loss": contrastive_loss,
            "regularization_loss": regularization_loss,
        }
