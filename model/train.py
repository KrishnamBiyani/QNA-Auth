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
from preprocessing.features import (
    FEATURE_VERSION,
    get_canonical_feature_names,
    NoisePreprocessor,
    FeatureVector,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> torch.Generator:
    """Set random seeds for reproducibility (torch, numpy, dataloader). Returns a generator for DataLoader."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    logger.info(f"Random seed set to {seed}")
    return g


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
            patience=5
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
        save_dir: str = "./model/checkpoints",
        save_last_n: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs. Saves best model by validation loss; optionally keeps last N checkpoints.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader (required for best-by-val saving)
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
            save_last_n: If set, keep only the last N epoch checkpoints (e.g. 3); older ones are removed.

        Returns:
            Training history dictionary
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        saved_epoch_paths: List[Path] = []

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.history["train_loss"].append(train_loss)
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")

            val_loss = None
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                logger.info(f"Epoch {epoch}/{epochs} - Val Loss: {val_loss:.4f}")
                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = save_path / "best_model.pt"
                    self.save_checkpoint(checkpoint_path, epoch, val_loss)
                    logger.info(f"Saved best model (val_loss={val_loss:.4f})")

            # Periodic checkpoint (last N only if save_last_n is set)
            if epoch % 10 == 0:
                ckpt_name = f"checkpoint_epoch_{epoch}.pt"
                checkpoint_path = save_path / ckpt_name
                self.save_checkpoint(
                    checkpoint_path,
                    epoch,
                    (val_loss if val_loss is not None else train_loss),
                )
                saved_epoch_paths.append(checkpoint_path)
                if save_last_n is not None and len(saved_epoch_paths) > save_last_n:
                    to_remove = saved_epoch_paths.pop(0)
                    if to_remove.exists():
                        to_remove.unlink()
                        logger.info(f"Removed old checkpoint {to_remove.name}")

        final_path = save_path / "final_model.pt"
        self.save_checkpoint(final_path, epochs, self.history["train_loss"][-1])
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


def load_real_dataset(
    data_dir: str = "./dataset/samples",
    max_samples: Optional[int] = 20,
):
    """Load and process real dataset from disk. Uses canonical feature names for train/serve consistency.

    Args:
        data_dir: Path to dataset/samples (contains json/ and raw .npy).
        max_samples: If set, use at most this many samples (for quick demos). None = use all.
    """
    print(f"Loading dataset from {data_dir}...")
    json_dir = Path(data_dir) / "json"
    if not json_dir.exists():
        print("Dataset directory not found!")
        return {}, 0

    preprocessor = NoisePreprocessor(normalize=True)
    converter = FeatureVector(get_canonical_feature_names())

    features_by_device = {}
    sample_count = 0

    json_files = list(json_dir.glob("*.json"))
    if max_samples is not None and len(json_files) > max_samples:
        import random
        random.shuffle(json_files)
        json_files = json_files[:max_samples]
        print(f"Subsampling to {max_samples} samples for speed...")
    print(f"Found {len(json_files)} samples.")
    
    if not json_files:
        return {}, 0
    
    for json_file in tqdm(json_files, desc="Processing samples"):
        try:
            with open(json_file, 'r') as f:
                meta = json.load(f)
            
            device_id = meta['device_id']
            # fix path (remove leading slash if present in metadata relative path)
            rel_path = meta['raw_data_path'].lstrip('/\\')
            raw_path = Path(data_dir) / rel_path
            
            if not raw_path.exists():
                # Fallback to absolute check or check if relative to root
                if (Path(data_dir).parent.parent / rel_path).exists():
                     raw_path = Path(data_dir).parent.parent / rel_path
                else:
                    # logger.warning(f"Raw file not found: {raw_path}")
                    continue
                
            raw_data = np.load(raw_path)
            
            # Process
            features = preprocessor.extract_all_features(raw_data)
            vector = converter.to_vector(features)
            
            if device_id not in features_by_device:
                features_by_device[device_id] = []
            
            features_by_device[device_id].append(vector)
            sample_count += 1
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            continue
            
    input_dim = len(next(iter(features_by_device.values()))[0]) if sample_count > 0 else 0
    logger.info(f"Loaded {sample_count} samples from {len(features_by_device)} devices")
    return features_by_device, input_dim


def main(seed: int = 42):
    """Train pipeline with real data. Uses canonical features and reproducible seeds."""
    print("\n=== Siamese Network Training ===")

    set_seed(seed)

    # Load Real Dataset (canonical feature names; max_samples=20 for quick demo)
    features_by_device, input_dim = load_real_dataset(max_samples=20)

    if not features_by_device or len(features_by_device) < 2:
        print("!! INSUFFICIENT DATA !!")
        print("Need at least 2 devices with samples to train.")
        print("Run scripts/collect_data_for_training.py and scripts/ingest_collected_data.py to add data.")
        return

    embedding_dim = 128

    print(f"Input Feature Dimension: {input_dim}")
    print(f"Devices: {len(features_by_device)}")

    total_samples = sum(len(v) for v in features_by_device.values())
    samples_per_epoch = max(100, total_samples * 2)

    train_dataset = TripletDataset(features_by_device, samples_per_epoch=samples_per_epoch)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(32, samples_per_epoch),
        shuffle=True,
        generator=g,
        worker_init_fn=lambda wid: np.random.seed(seed + wid) if hasattr(np.random, "seed") else None,
    )

    model = SiameseNetwork(input_dim=input_dim, embedding_dim=embedding_dim)

    trainer = ModelTrainer(
        model=model,
        loss_type="triplet",
        learning_rate=0.001,
    )

    print("\n=== Starting Training Loop ===")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=None,
        epochs=10,
        save_dir="./model/checkpoints",
        save_last_n=3,
    )
    
    print("\n=== Training Completed ===")
    print(f"Final Loss: {history['train_loss'][-1]:.4f}")

    # Save server-style checkpoint (with feature_names + feature_version) for train/serve consistency
    best_path = Path("model/checkpoints/best_model.pt")
    server_path = Path("server/models/best_model.pt")
    server_path.parent.mkdir(parents=True, exist_ok=True)
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        server_ckpt = {
            "model_state_dict": ckpt["model_state_dict"],
            "embedding_dim": embedding_dim,
            "input_dim": input_dim,
            "feature_names": get_canonical_feature_names(),
            "feature_version": FEATURE_VERSION,
        }
        torch.save(server_ckpt, server_path)
        print(f"Deployed model (with feature pipeline v{FEATURE_VERSION}) to {server_path}")
    else:
        print("No best_model.pt found; skipping server deploy.")


if __name__ == "__main__":
    main()
