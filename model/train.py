"""
Training Script for Siamese Network
Trains the model using triplet or contrastive loss
"""

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import json
import logging
from tqdm import tqdm

from .siamese_model import SiameseNetwork, TripletLoss, ContrastiveLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """Dataset for triplet training"""
    
    def __init__(
        self,
        features_by_device: Dict[str, List[np.ndarray]],
        samples_per_epoch: int = 1000
    ):
        """
        Initialize triplet dataset
        
        Args:
            features_by_device: Dictionary mapping device_id to list of feature arrays
            samples_per_epoch: Number of triplets to generate per epoch
        """
        self.features_by_device = features_by_device
        self.device_ids = list(features_by_device.keys())
        self.samples_per_epoch = samples_per_epoch
        
        logger.info(f"TripletDataset: {len(self.device_ids)} devices, "
                   f"{samples_per_epoch} triplets/epoch")
    
    def __len__(self) -> int:
        return self.samples_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a triplet (anchor, positive, negative)
        
        Returns:
            Tuple of (anchor, positive, negative) tensors
        """
        # Select anchor device
        anchor_device = np.random.choice(self.device_ids)
        anchor_features = self.features_by_device[anchor_device]
        
        # Select anchor and positive from same device
        anchor_idx, positive_idx = np.random.choice(
            len(anchor_features), size=2, replace=True
        )
        anchor = anchor_features[anchor_idx]
        positive = anchor_features[positive_idx]
        
        # Select negative from different device
        negative_device = np.random.choice(
            [d for d in self.device_ids if d != anchor_device]
        )
        negative_features = self.features_by_device[negative_device]
        negative_idx = np.random.choice(len(negative_features))
        negative = negative_features[negative_idx]
        
        # Convert to tensors
        anchor = torch.from_numpy(anchor).float()
        positive = torch.from_numpy(positive).float()
        negative = torch.from_numpy(negative).float()
        
        return anchor, positive, negative


class PairDataset(Dataset):
    """Dataset for contrastive learning with pairs"""
    
    def __init__(
        self,
        features_by_device: Dict[str, List[np.ndarray]],
        samples_per_epoch: int = 1000,
        positive_ratio: float = 0.5
    ):
        """
        Initialize pair dataset
        
        Args:
            features_by_device: Dictionary mapping device_id to list of feature arrays
            samples_per_epoch: Number of pairs to generate per epoch
            positive_ratio: Ratio of positive pairs (same device)
        """
        self.features_by_device = features_by_device
        self.device_ids = list(features_by_device.keys())
        self.samples_per_epoch = samples_per_epoch
        self.positive_ratio = positive_ratio
        
        logger.info(f"PairDataset: {len(self.device_ids)} devices, "
                   f"{samples_per_epoch} pairs/epoch")
    
    def __len__(self) -> int:
        return self.samples_per_epoch
    
    def __getitem__(
        self, 
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a pair with label
        
        Returns:
            Tuple of (sample1, sample2, label) where label=1 for same device
        """
        # Decide if positive or negative pair
        is_positive = np.random.random() < self.positive_ratio
        
        if is_positive:
            # Same device
            device = np.random.choice(self.device_ids)
            features = self.features_by_device[device]
            idx1, idx2 = np.random.choice(len(features), size=2, replace=True)
            sample1 = features[idx1]
            sample2 = features[idx2]
            label = 1.0
        else:
            # Different devices
            device1, device2 = np.random.choice(self.device_ids, size=2, replace=False)
            features1 = self.features_by_device[device1]
            features2 = self.features_by_device[device2]
            sample1 = features1[np.random.choice(len(features1))]
            sample2 = features2[np.random.choice(len(features2))]
            label = 0.0
        
        # Convert to tensors
        sample1 = torch.from_numpy(sample1).float()
        sample2 = torch.from_numpy(sample2).float()
        label = torch.tensor(label).float()
        
        return sample1, sample2, label


class ModelTrainer:
    """Trains Siamese network for device authentication"""
    
    def __init__(
        self,
        model: SiameseNetwork,
        loss_type: str = 'triplet',
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer
        
        Args:
            model: SiameseNetwork model
            loss_type: 'triplet' or 'contrastive'
            learning_rate: Learning rate
            device: Device to train on
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Loss function
        if loss_type == 'triplet':
            self.criterion = TripletLoss(margin=1.0)
        elif loss_type == 'contrastive':
            self.criterion = ContrastiveLoss(margin=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.loss_type = loss_type
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        logger.info(f"ModelTrainer initialized on {self.device}")
        logger.info(f"Loss type: {loss_type}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            if self.loss_type == 'triplet':
                anchor, positive, negative = batch
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Forward pass
                anchor_emb, pos_emb, neg_emb = self.model(anchor, positive, negative)
                
                # Compute loss
                loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                
            elif self.loss_type == 'contrastive':
                sample1, sample2, label = batch
                sample1 = sample1.to(self.device)
                sample2 = sample2.to(self.device)
                label = label.to(self.device)
                
                # Forward pass
                emb1 = self.model.forward_one(sample1)
                emb2 = self.model.forward_one(sample2)
                
                # Compute loss
                loss = self.criterion(emb1, emb2, label)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                if self.loss_type == 'triplet':
                    anchor, positive, negative = batch
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)
                    
                    anchor_emb, pos_emb, neg_emb = self.model(anchor, positive, negative)
                    loss = self.criterion(anchor_emb, pos_emb, neg_emb)
                    
                elif self.loss_type == 'contrastive':
                    sample1, sample2, label = batch
                    sample1 = sample1.to(self.device)
                    sample2 = sample2.to(self.device)
                    label = label.to(self.device)
                    
                    emb1 = self.model.forward_one(sample1)
                    emb2 = self.model.forward_one(sample2)
                    loss = self.criterion(emb1, emb2, label)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        save_dir: str = "./model/checkpoints"
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                logger.info(f"Epoch {epoch}/{epochs} - Val Loss: {val_loss:.4f}")
                
                # Update learning rate
                self.scheduler.step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = save_path / "best_model.pt"
                    self.save_checkpoint(checkpoint_path, epoch, val_loss)
                    logger.info(f"Saved best model (val_loss={val_loss:.4f})")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                checkpoint_path = save_path / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(
                    checkpoint_path,
                    epoch,
                    self.history['val_loss'][-1] if val_loader else train_loss
                )
        
        # Save final model
        final_path = save_path / "final_model.pt"
        self.save_checkpoint(final_path, epochs, train_loss)
        logger.info("Training completed!")
        
        return self.history
    
    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        loss: float
    ):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def main():
    """Test training pipeline"""
    print("\n=== Training Pipeline Test ===")
    
    # Create synthetic dataset
    input_dim = 50
    embedding_dim = 128
    num_devices = 5
    samples_per_device = 50
    
    # Generate synthetic features for each device
    features_by_device = {}
    for i in range(num_devices):
        device_id = f"device_{i:03d}"
        # Each device has slightly different feature distribution
        base_features = np.random.randn(samples_per_device, input_dim) + i * 0.5
        features_by_device[device_id] = [base_features[j] for j in range(samples_per_device)]
    
    print(f"Created dataset with {num_devices} devices")
    
    # Create datasets
    train_dataset = TripletDataset(features_by_device, samples_per_epoch=200)
    val_dataset = TripletDataset(features_by_device, samples_per_epoch=50)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        loss_type='triplet',
        learning_rate=0.001
    )
    
    # Train
    print("\n=== Training ===")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        save_dir="./model/test_checkpoints"
    )
    
    print("\n=== Training History ===")
    print(f"Train losses: {[f'{l:.4f}' for l in history['train_loss']]}")
    print(f"Val losses: {[f'{l:.4f}' for l in history['val_loss']]}")


if __name__ == "__main__":
    main()
