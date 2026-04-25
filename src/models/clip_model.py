"""CLIP-based image-text matching model."""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class CLIPImageTextMatcher(nn.Module):
    """
    CLIP-based image-text matching model with enhanced features.
    
    Supports fine-tuning, adapter layers, and various similarity metrics.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze_vision_encoder: bool = False,
        freeze_text_encoder: bool = False,
        use_adapter: bool = False,
        adapter_dim: int = 64,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        dropout: float = 0.1,
        vision_dropout: float = 0.1,
        text_dropout: float = 0.1,
    ):
        """
        Initialize the CLIP image-text matcher.
        
        Args:
            model_name: CLIP model name
            freeze_vision_encoder: Whether to freeze vision encoder
            freeze_text_encoder: Whether to freeze text encoder
            use_adapter: Whether to use adapter layers
            adapter_dim: Adapter dimension
            temperature: Temperature for similarity computation
            learnable_temperature: Whether temperature is learnable
            dropout: General dropout rate
            vision_dropout: Vision encoder dropout
            text_dropout: Text encoder dropout
        """
        super().__init__()
        
        self.model_name = model_name
        self.temperature = temperature
        self.learnable_temperature = learnable_temperature
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        
        # Freeze encoders if specified
        if freeze_vision_encoder:
            self._freeze_encoder(self.clip_model.vision_model)
        
        if freeze_text_encoder:
            self._freeze_encoder(self.clip_model.text_model)
        
        # Add dropout layers
        self.vision_dropout = nn.Dropout(vision_dropout)
        self.text_dropout = nn.Dropout(text_dropout)
        
        # Adapter layers
        self.use_adapter = use_adapter
        if use_adapter:
            self._setup_adapters(adapter_dim)
        
        # Learnable temperature
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))
        else:
            self.register_buffer("logit_scale", torch.tensor(math.log(1 / temperature)))
        
        # Additional projection layers for enhanced features
        self.vision_projection = nn.Linear(
            self.clip_model.config.vision_config.hidden_size,
            self.clip_model.config.projection_dim,
        )
        self.text_projection = nn.Linear(
            self.clip_model.config.text_config.hidden_size,
            self.clip_model.config.projection_dim,
        )
        
        # Initialize projections with CLIP weights
        self._initialize_projections()
    
    def _freeze_encoder(self, encoder: nn.Module) -> None:
        """Freeze encoder parameters."""
        for param in encoder.parameters():
            param.requires_grad = False
    
    def _setup_adapters(self, adapter_dim: int) -> None:
        """Setup adapter layers for parameter-efficient fine-tuning."""
        vision_hidden_size = self.clip_model.config.vision_config.hidden_size
        text_hidden_size = self.clip_model.config.text_config.hidden_size
        
        # Vision adapter
        self.vision_adapter_down = nn.Linear(vision_hidden_size, adapter_dim)
        self.vision_adapter_up = nn.Linear(adapter_dim, vision_hidden_size)
        
        # Text adapter
        self.text_adapter_down = nn.Linear(text_hidden_size, adapter_dim)
        self.text_adapter_up = nn.Linear(adapter_dim, text_hidden_size)
        
        # Initialize adapters
        nn.init.xavier_uniform_(self.vision_adapter_down.weight)
        nn.init.zeros_(self.vision_adapter_down.bias)
        nn.init.xavier_uniform_(self.vision_adapter_up.weight)
        nn.init.zeros_(self.vision_adapter_up.bias)
        
        nn.init.xavier_uniform_(self.text_adapter_down.weight)
        nn.init.zeros_(self.text_adapter_down.bias)
        nn.init.xavier_uniform_(self.text_adapter_up.weight)
        nn.init.zeros_(self.text_adapter_up.bias)
    
    def _initialize_projections(self) -> None:
        """Initialize projection layers with CLIP weights."""
        with torch.no_grad():
            self.vision_projection.weight.copy_(
                self.clip_model.visual_projection.weight
            )
            self.vision_projection.bias.copy_(
                self.clip_model.visual_projection.bias
            )
            
            self.text_projection.weight.copy_(
                self.clip_model.text_projection.weight
            )
            self.text_projection.bias.copy_(
                self.clip_model.text_projection.bias
            )
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            pixel_values: Preprocessed image tensors
            
        Returns:
            Image embeddings
        """
        # Get vision features
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state
        
        # Apply adapter if enabled
        if self.use_adapter:
            vision_features = self._apply_vision_adapter(vision_features)
        
        # Apply dropout
        vision_features = self.vision_dropout(vision_features)
        
        # Pool features (use CLIP's pooling)
        pooled_vision_features = vision_features[:, 0, :]  # CLS token
        
        # Project to embedding space
        image_embeds = self.vision_projection(pooled_vision_features)
        
        return image_embeds
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            input_ids: Tokenized text input
            attention_mask: Attention mask for text
            
        Returns:
            Text embeddings
        """
        # Get text features
        text_outputs = self.clip_model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = text_outputs.last_hidden_state
        
        # Apply adapter if enabled
        if self.use_adapter:
            text_features = self._apply_text_adapter(text_features)
        
        # Apply dropout
        text_features = self.text_dropout(text_features)
        
        # Pool features (use CLIP's pooling)
        pooled_text_features = text_features[
            torch.arange(text_features.shape[0]), 
            input_ids.argmax(dim=-1)
        ]
        
        # Project to embedding space
        text_embeds = self.text_projection(pooled_text_features)
        
        return text_embeds
    
    def _apply_vision_adapter(self, features: torch.Tensor) -> torch.Tensor:
        """Apply vision adapter layers."""
        # Down-projection
        down_features = self.vision_adapter_down(features)
        down_features = F.relu(down_features)
        
        # Up-projection
        up_features = self.vision_adapter_up(down_features)
        
        # Residual connection
        return features + up_features
    
    def _apply_text_adapter(self, features: torch.Tensor) -> torch.Tensor:
        """Apply text adapter layers."""
        # Down-projection
        down_features = self.text_adapter_down(features)
        down_features = F.relu(down_features)
        
        # Up-projection
        up_features = self.text_adapter_up(down_features)
        
        # Residual connection
        return features + up_features
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for image-text matching.
        
        Args:
            pixel_values: Preprocessed image tensors
            input_ids: Tokenized text input
            attention_mask: Attention mask for text
            
        Returns:
            Dictionary containing embeddings and similarity scores
        """
        # Encode images and text
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        
        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # Compute similarity scores
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "logit_scale": logit_scale,
        }
    
    def compute_similarity(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        metric: str = "cosine",
    ) -> torch.Tensor:
        """
        Compute similarity between image and text embeddings.
        
        Args:
            image_embeds: Image embeddings
            text_embeds: Text embeddings
            metric: Similarity metric ('cosine', 'dot_product', 'euclidean')
            
        Returns:
            Similarity scores
        """
        if metric == "cosine":
            return F.cosine_similarity(image_embeds, text_embeds, dim=-1)
        elif metric == "dot_product":
            return torch.sum(image_embeds * text_embeds, dim=-1)
        elif metric == "euclidean":
            return -torch.norm(image_embeds - text_embeds, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def get_attention_weights(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for visualization.
        
        Args:
            pixel_values: Preprocessed image tensors
            input_ids: Tokenized text input
            attention_mask: Attention mask for text
            
        Returns:
            Dictionary containing attention weights
        """
        with torch.no_grad():
            # Get vision attention
            vision_outputs = self.clip_model.vision_model(
                pixel_values=pixel_values, 
                output_attentions=True
            )
            vision_attentions = vision_outputs.attentions
            
            # Get text attention
            text_outputs = self.clip_model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            text_attentions = text_outputs.attentions
        
        return {
            "vision_attentions": vision_attentions,
            "text_attentions": text_attentions,
        }
    
    def get_processor(self) -> CLIPProcessor:
        """Get CLIP processor for preprocessing."""
        return CLIPProcessor.from_pretrained(self.model_name)
