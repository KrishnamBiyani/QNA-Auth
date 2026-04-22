# Noise-Based Device Verification Configuration
# Copy this file to config.py and adjust values

import os

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
    "similarity_threshold": 0.97,
    "similarity_metric": "cosine",  # or "euclidean"
    "max_auth_attempts": 3
}

AUTH_CONFIDENCE_STRONG = 0.97
AUTH_CONFIDENCE_UNCERTAIN = 0.92
SIMILARITY_THRESHOLD = AUTH_CONFIDENCE_STRONG
AUTH_PROFILE_GUARD_Z = 6.0
AUTH_PROFILE_GUARD_MIN_DELTA = 0.02
AUTH_IDENTIFICATION_MARGIN = 0.02
AUTH_DRIFT_EMA_ALPHA = 0.2
AUTH_DRIFT_UPDATE_ENABLED = True
AUTH_DRIFT_MIN_STRONG_MATCHES = 2
AUTH_SOURCE_WEIGHTS = {
    "camera": 0.7,
    "microphone": 0.3,
}

# Challenge-Response Configuration
CHALLENGE_CONFIG = {
    "nonce_length": 32,
    "challenge_expiry_seconds": 60,
    "anti_replay_window_seconds": 300
}
CHALLENGE_SERVER_SECRET = os.environ.get(
    "QNA_AUTH_SERVER_SECRET",
    "dev-only-qna-auth-server-secret-change-me",
)

# Noise Collection Configuration
NOISE_CONFIG = {
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

# Canonical preprocessing mode used across train/eval/server runtime.
PREPROCESSING_NORMALIZE = PREPROCESSING_CONFIG["normalize"]

# Storage Configuration
STORAGE_CONFIG = {
    "dataset_dir": "./dataset/samples",
    "embeddings_dir": "./auth/device_embeddings",
    "checkpoints_dir": "./model/checkpoints"
}

# Database (SQLite by default; use DATABASE_URL for PostgreSQL etc.)
# Example: "sqlite:///./data/qna_auth.db"
DATABASE_URL = "sqlite:///./data/qna_auth.db"

# Server Configuration
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,  # Set to True for development
    "workers": 1
}

# CORS Configuration (use these origins in production; no "*" with credentials)
CORS_CONFIG = {
    "allow_origins": ["http://localhost:3000", "http://localhost:5173"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"]
}

# API protection (optional)
# Set API_KEY to a secret string to require X-API-Key header on enroll/authenticate/challenge/verify/delete
API_KEY = None  # e.g. os.environ.get("QNA_AUTH_API_KEY")

# Rate limiting for auth endpoints (per IP)
# RATE_LIMIT_REQUESTS = 0 disables rate limiting
RATE_LIMIT_REQUESTS = 30   # Max requests per window per IP
RATE_LIMIT_WINDOW_SEC = 60

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "qna_auth.log"
}
