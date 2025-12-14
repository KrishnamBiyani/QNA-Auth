# QNA-Auth Quick Start Guide

## First-Time Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Test noise collection
python -m noise_collection.qrng_api

# Should output quantum noise statistics
```

### 3. Start Backend Server

```bash
python server/app.py
```

Visit http://localhost:8000/docs for API documentation.

### 4. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000

## Common Tasks

### Enroll a New Device

1. Open http://localhost:3000/enroll
2. Choose noise sources (QRNG recommended)
3. Click "Enroll Device"
4. Copy the device ID

### Authenticate a Device

1. Open http://localhost:3000/authenticate
2. Select device from dropdown
3. Click "Authenticate"
4. View results

### View Enrolled Devices

1. Open http://localhost:3000/devices
2. Click on a device to see details
3. Delete devices if needed

## Training Custom Model

### Step 1: Collect Training Data

```python
from dataset.builder import DatasetBuilder
from noise_collection import QRNGClient

builder = DatasetBuilder()
qrng = QRNGClient()

# Enroll multiple devices
for i in range(5):
    device_id = f"device_{i:03d}"
    samples = qrng.fetch_multiple_samples(50)
    builder.add_batch(device_id, 'qrng', samples)
```

### Step 2: Process Features

```python
from preprocessing import NoisePreprocessor, FeatureVector

preprocessor = NoisePreprocessor()
converter = FeatureVector()

# Extract features from dataset
# (See model/train.py for complete example)
```

### Step 3: Train Model

```python
from model import SiameseNetwork, ModelTrainer

model = SiameseNetwork(input_dim=50, embedding_dim=128)
trainer = ModelTrainer(model, loss_type='triplet')
trainer.train(train_loader, val_loader, epochs=50)
```

### Step 4: Use Trained Model

```python
# Copy trained model to server directory
cp model/checkpoints/best_model.pt server/models/

# Restart server to load new model
```

## Troubleshooting

### Camera Not Working

- Ensure webcam is connected
- Check camera permissions
- Try different camera_index in config

### Microphone Not Working

- Check audio input permissions
- Verify microphone is not muted
- Install sounddevice: `pip install sounddevice`

### QRNG API Fails

- Check internet connection
- API may have rate limits (wait and retry)
- Use alternative: `qrng_client = QRNGClient(api_url=...)`

### Low Authentication Accuracy

- Increase number of samples during enrollment
- Use multiple noise sources
- Train model on more diverse data
- Adjust similarity threshold

## Development Tips

### Running Tests

```bash
# Test individual modules
python -m pytest tests/

# Or test manually
python -m noise_collection.qrng_api
python -m preprocessing.features
python -m model.siamese_model
```

### Code Formatting

```bash
# Format Python code
black .

# Lint
flake8 .
```

### Frontend Development

```bash
cd frontend

# Type checking
npm run build

# Linting
npm run lint
```

## Production Deployment

### Security Checklist

- [ ] Change CORS origins in server/app.py
- [ ] Add authentication/API keys for endpoints
- [ ] Enable HTTPS
- [ ] Encrypt device embeddings at rest
- [ ] Set up proper logging
- [ ] Configure rate limiting
- [ ] Review security settings in config.py

### Performance Optimization

- [ ] Use ONNX runtime for inference
- [ ] Enable model quantization
- [ ] Add caching layer
- [ ] Use production WSGI server (gunicorn)
- [ ] Optimize frontend build (npm run build)
- [ ] Enable gzip compression

## Resources

- API Documentation: http://localhost:8000/docs
- Frontend: http://localhost:3000
- ANU QRNG: https://qrng.anu.edu.au/
- PyTorch Docs: https://pytorch.org/docs/
- FastAPI Docs: https://fastapi.tiangolo.com/
