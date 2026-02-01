import os
import shutil
import torch
from dataset.builder import DatasetBuilder
from model.siamese_model import SiameseNetwork
# from model.train import ModelTrainer # Importing locally to avoid circular issues if any
import config

def main():
    print("--- 1. BUILDING DATASET ---")
    builder = DatasetBuilder()
    
    # Check if we have raw data
    if not os.path.exists(config.DATA_DIR) or len(os.listdir(config.DATA_DIR)) == 0:
        print("!! NO RAW DATA FOUND !!")
        print("Tip: Run 'python collect_training_data.py' to collect real samples.")
        print("Continuing with DUMMY/INITIALIZED model for server testing...")
    else:
        # Process raw data into CSVs
        try:
            builder.export_for_training(config.PROCESSED_DIR)
            print(f"Dataset exported to {config.PROCESSED_DIR}")
        except Exception as e:
            print(f"Dataset export warning: {e}")

    print("\n--- 2. INITIALIZING MODEL ---")
    # Initialize the model structure
    # Detect actual feature dimension by extracting from dummy sample
    from preprocessing.features import NoisePreprocessor, FeatureVector
    import numpy as np
    
    preprocessor = NoisePreprocessor(normalize=False)
    converter = FeatureVector()
    dummy_sample = np.random.randn(1024)
    dummy_features = preprocessor.extract_all_features(dummy_sample)
    dummy_vector = converter.to_vector(dummy_features)
    input_dim = len(dummy_vector)
    
    print(f"Detected feature dimension: {input_dim}")
    
    # Initialize model with correct dimensions
    model = SiameseNetwork(input_dim=input_dim, embedding_dim=128).to(config.DEVICE)
    
    print("\n--- 3. TRAINING LOOP (Skipped for Setup) ---")
    print("Skipping actual training loop to generate initial model file.")
    print("To train for real, you will need to collect data first.")

    print("\n--- 4. SAVING & DEPLOYING MODEL ---")
    # 1. Save locally in model/checkpoints
    os.makedirs("model/checkpoints", exist_ok=True)
    save_path = "model/checkpoints/best_model.pt"
    # Save in the format expected by DeviceEmbedder.load_model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 0,
        'loss': 0.0,
        'input_dim': input_dim,
        'embedding_dim': 128
    }
    torch.save(checkpoint, save_path)
    print(f"Model saved locally to {save_path}")

    # 2. Deploy to Server folder
    server_model_path = config.MODEL_PATH
    server_model_dir = os.path.dirname(server_model_path)
    os.makedirs(server_model_dir, exist_ok=True)
    
    shutil.copy(save_path, server_model_path)
    print(f"Model deployed to Server at: {server_model_path}")
    print("You can now run 'python server/app.py' safely!")

if __name__ == "__main__":
    main()
