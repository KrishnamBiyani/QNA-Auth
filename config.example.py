# QNA-Auth Configuration
# Copy this file to config.py and adjust values

# Model Configuration
MODEL_CONFIG = {
    "input_dim": 50,  # Must match feature extractor output
    "embedding_dim": 128,
    "hidden_dims": [256, 256, 128],
    "model_path": "./model/checkpoints/best_model.pt"
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50,
    "loss_type": "triplet",  # or "contrastive"
    "margin": 1.0,
    "samples_per_epoch": 1000
}

# Authentication Configuration
AUTH_CONFIG = {
    "similarity_threshold": 0.85,
    "similarity_metric": "cosine",  # or "euclidean"
    "max_auth_attempts": 3
}

# Challenge-Response Configuration
CHALLENGE_CONFIG = {
    "nonce_length": 32,
    "challenge_expiry_seconds": 60,
    "anti_replay_window_seconds": 300
}

# Noise Collection Configuration
NOISE_CONFIG = {
    "qrng": {
        "api_url": "https://qrng.anu.edu.au/API/jsonI.php",
        "sample_size": 1024,
        "num_samples": 50
    },
    "camera": {
        "camera_index": 0,
        "exposure_time": 0.1,
        "num_frames": 50
    },
    "microphone": {
        "sample_rate": 44100,
        "duration": 0.5,
        "num_samples": 50
    }
}

# Preprocessing Configuration
PREPROCESSING_CONFIG = {
    "normalize": True,
    "normalization_method": "standard",  # "standard", "minmax", or "robust"
    "fft_sample_rate": 1.0,
    "entropy_bins": 256
}

# Storage Configuration
STORAGE_CONFIG = {
    "dataset_dir": "./dataset/samples",
    "embeddings_dir": "./auth/device_embeddings",
    "checkpoints_dir": "./model/checkpoints"
}

# Server Configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,  # Set to True for development
    "workers": 1
}

# CORS Configuration
CORS_CONFIG = {
    "allow_origins": ["http://localhost:3000", "http://localhost:5173"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"]
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "qna_auth.log"
}
