"""
Siamese Neural Network for Device Authentication
Creates unique embeddings from noise samples using contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingNetwork(nn.Module):
    """Base embedding network that processes input features"""
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: list = [256, 256, 128]
    ):
        """
        Initialize embedding network
        
        Args:
            input_dim: Dimension of input features
            embedding_dim: Dimension of output embedding
            hidden_dims: List of hidden layer dimensions
        """
        super(EmbeddingNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                # nn.BatchNorm1d(hidden_dim),  # Disabled due to CUDA device issues
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.network = nn.Sequential(*layers)
        
        # L2 normalization layer
        self.normalize = nn.functional.normalize
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Normalized embeddings [batch_size, embedding_dim]
        """
        embedding = self.network(x)
        # L2 normalize embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class SiameseNetwork(nn.Module):
    """Siamese network with shared embedding network"""
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dims: list = [256, 256, 128]
    ):
        """
        Initialize Siamese network
        
        Args:
            input_dim: Dimension of input features
            embedding_dim: Dimension of output embedding
            hidden_dims: List of hidden layer dimensions
        """
        super(SiameseNetwork, self).__init__()
        
        # Shared embedding network
        self.embedding_network = EmbeddingNetwork(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims
        )
        
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process single input
        
        Args:
            x: Input features
            
        Returns:
            Embedding
        """
        return self.embedding_network(x)
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for triplet
        
        Args:
            anchor: Anchor samples
            positive: Positive samples (same device)
            negative: Negative samples (different device)
            
        Returns:
            Tuple of (anchor_embedding, positive_embedding, negative_embedding)
        """
        anchor_emb = self.forward_one(anchor)
        positive_emb = self.forward_one(positive)
        negative_emb = self.forward_one(negative)
        
        return anchor_emb, positive_emb, negative_emb


class TripletLoss(nn.Module):
    """Triplet loss for metric learning"""
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize triplet loss
        
        Args:
            margin: Margin for triplet loss
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings
            
        Returns:
            Loss value
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks"""
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss
        
        Args:
            margin: Margin for negative pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss
        
        Args:
            embedding1: First embeddings
            embedding2: Second embeddings
            label: 1 for same device, 0 for different devices
            
        Returns:
            Loss value
        """
        # Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Contrastive loss
        loss_positive = label * torch.pow(distance, 2)
        loss_negative = (1 - label) * torch.pow(F.relu(self.margin - distance), 2)
        
        loss = 0.5 * (loss_positive + loss_negative)
        
        return loss.mean()


class DeviceEmbedder:
    """High-level interface for device embedding"""
    
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize device embedder
        
        Args:
            input_dim: Input feature dimension
            embedding_dim: Embedding dimension
            device: Device to run model on
        """
        # Explicitly use cuda:0 to avoid device mismatch
        self.device = torch.device(device)
        self.model = SiameseNetwork(
            input_dim=input_dim,
            embedding_dim=embedding_dim
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set to eval mode
        self.model.eval()
        
        self.embedding_dim = embedding_dim
        
        logger.info(f"Initialized DeviceEmbedder on {self.device}")
        logger.info(f"Input dim: {input_dim}, Embedding dim: {embedding_dim}")
    
    def embed(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate embedding for features
        
        Args:
            features: Input features [batch_size, input_dim] or [input_dim]
            
        Returns:
            Embeddings [batch_size, embedding_dim] or [embedding_dim]
        """
        self.model.eval()
        
        # Handle single sample
        if features.dim() == 1:
            features = features.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Ensure features is float32 and on the correct device
        features = features.float().to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model.forward_one(features)
        
        # Return to CPU
        embedding = embedding.cpu()
        
        # Return single embedding if input was single sample
        if single_sample:
            embedding = embedding.squeeze(0)
        
        return embedding
    
    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
        metric: str = 'cosine'
    ) -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            # Cosine similarity (higher = more similar)
            similarity = F.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            ).item()
        elif metric == 'euclidean':
            # Euclidean distance (lower = more similar)
            similarity = -F.pairwise_distance(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0),
                p=2
            ).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarity
    
    def save_model(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'input_dim': self.model.embedding_network.input_dim
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Model loaded from {path}")


def main():
    """Test Siamese network"""
    print("\n=== Siamese Neural Network Test ===")
    
    # Parameters
    input_dim = 50
    embedding_dim = 128
    batch_size = 32
    
    # Create model
    model = SiameseNetwork(
        input_dim=input_dim,
        embedding_dim=embedding_dim
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    anchor = torch.randn(batch_size, input_dim)
    positive = torch.randn(batch_size, input_dim)
    negative = torch.randn(batch_size, input_dim)
    
    anchor_emb, pos_emb, neg_emb = model(anchor, positive, negative)
    
    print(f"Anchor embedding shape: {anchor_emb.shape}")
    print(f"Positive embedding shape: {pos_emb.shape}")
    print(f"Negative embedding shape: {neg_emb.shape}")
    
    # Test losses
    print("\n=== Testing Triplet Loss ===")
    triplet_loss = TripletLoss(margin=1.0)
    loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
    print(f"Triplet loss: {loss.item():.4f}")
    
    print("\n=== Testing Contrastive Loss ===")
    contrastive_loss = ContrastiveLoss(margin=1.0)
    labels = torch.ones(batch_size)  # Same device
    loss = contrastive_loss(anchor_emb, pos_emb, labels)
    print(f"Contrastive loss (positive pairs): {loss.item():.4f}")
    
    labels = torch.zeros(batch_size)  # Different devices
    loss = contrastive_loss(anchor_emb, neg_emb, labels)
    print(f"Contrastive loss (negative pairs): {loss.item():.4f}")
    
    # Test embedder
    print("\n=== Testing DeviceEmbedder ===")
    embedder = DeviceEmbedder(input_dim=input_dim, embedding_dim=embedding_dim)
    
    # Single sample
    single_features = torch.randn(input_dim)
    embedding = embedder.embed(single_features)
    print(f"Single embedding shape: {embedding.shape}")
    
    # Batch
    batch_features = torch.randn(5, input_dim)
    embeddings = embedder.embed(batch_features)
    print(f"Batch embeddings shape: {embeddings.shape}")
    
    # Similarity
    emb1 = embedder.embed(torch.randn(input_dim))
    emb2 = embedder.embed(torch.randn(input_dim))
    sim = embedder.compute_similarity(emb1, emb2, metric='cosine')
    print(f"Cosine similarity: {sim:.4f}")


if __name__ == "__main__":
    main()
