"""
FastAPI Backend Server for QNA-Auth
Implements REST API for device enrollment and authentication
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running `python server/app.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import numpy as np
import logging
import socket

# Import QNA-Auth modules
from model.siamese_model import SiameseNetwork, DeviceEmbedder
from dataset.builder import DatasetBuilder
from preprocessing.features import NoisePreprocessor, FeatureVector, get_canonical_feature_names
from auth import (
    DeviceEnroller,
    DeviceAuthenticator,
    ChallengeResponseProtocol,
    SecureAuthenticationFlow,
    AuthenticationSession
)
import config  # Import configuration
from db import init_db
from db.models import Device
from db.session import get_session_factory
from db.challenge_store import DbChallengeStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="QNA-Auth API",
    description="Quantum Noise Assisted Authentication System",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"status": "online", "message": "QNA-Auth Server is running"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use proper dependency injection)
class AppState:
    embedder: Optional[DeviceEmbedder] = None
    preprocessor: Optional[NoisePreprocessor] = None
    feature_converter: Optional[FeatureVector] = None
    enroller: Optional[DeviceEnroller] = None
    authenticator: Optional[DeviceAuthenticator] = None
    challenge_protocol: Optional[ChallengeResponseProtocol] = None
    auth_flow: Optional[SecureAuthenticationFlow] = None

state = AppState()


# Pydantic models
class EnrollmentRequest(BaseModel):
    device_name: Optional[str] = Field(None, description="Optional device name")
    num_samples: int = Field(50, description="Number of noise samples to collect", ge=10, le=200)
    sources: List[str] = Field(['qrng'], description="Noise sources to use")
    client_samples: Optional[Dict[str, List[List[float]]]] = Field(None, description="Optional raw noise samples provided by client")


class EnrollmentResponse(BaseModel):
    device_id: str
    status: str
    message: str
    metadata: Dict


class AuthenticationRequest(BaseModel):
    device_id: str = Field(..., description="Device identifier to authenticate")
    sources: List[str] = Field(['qrng'], description="Noise sources to use")
    num_samples_per_source: int = Field(5, description="Samples per source", ge=1, le=20)
    client_samples: Optional[Dict[str, List[List[float]]]] = Field(None, description="Optional raw noise samples provided by client")


class ChallengeRequest(BaseModel):
    device_id: str = Field(..., description="Device identifier")


class ChallengeResponse(BaseModel):
    challenge_id: str
    nonce: str
    expires_at: str


class VerifyRequest(BaseModel):
    challenge_id: str = Field(..., description="Challenge identifier")
    response: str = Field(..., description="Challenge response signature")
    device_id: str = Field(..., description="Device identifier")
    noise_samples: List[List[float]] = Field(..., description="Fresh noise samples for embedding")


class VerifyResponse(BaseModel):
    authenticated: bool
    similarity: float
    details: Dict


class DeviceSummary(BaseModel):
    device_id: str
    device_name: Optional[str] = None


class DeviceListResponse(BaseModel):
    devices: List[DeviceSummary]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    components_initialized: bool
    db_ok: Optional[bool] = None  # True if DB is reachable, False on error
    devices_in_db: Optional[int] = None  # Number of devices in DB (when db_ok is True)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system components"""
    logger.info("Initializing QNA-Auth system...")
    
    try:
        # Initialize preprocessor first to determine feature dimension
        # DISABLE normalization to preserve amplitude/loudness differences between sensors
        state.preprocessor = NoisePreprocessor(normalize=False)
        # Use canonical feature list so train/serve use same feature order (overridden below if checkpoint has feature_names)
        state.feature_converter = FeatureVector(get_canonical_feature_names())
        
        # Extract features from a dummy sample to get the actual feature dimension
        dummy_sample = np.random.randn(1024)
        dummy_features = state.preprocessor.extract_all_features(dummy_sample)
        dummy_vector = state.feature_converter.to_vector(dummy_features)
        input_dim = len(dummy_vector)
        
        logger.info(f"Detected feature dimension: {input_dim}")
        
        embedding_dim = 128
        
        # Create embedder (this creates and initializes the model on device)
        state.embedder = DeviceEmbedder(input_dim=input_dim, embedding_dim=embedding_dim, device=config.DEVICE)
        
        # Load model if checkpoint exists; use feature_names from checkpoint for train/serve consistency
        model_path = Path(config.MODEL_PATH)
        
        if model_path.exists():
            extra = state.embedder.load_model(str(model_path))
            if extra.get("feature_names") is not None:
                state.feature_converter = FeatureVector(extra["feature_names"])
                logger.info(f"Using feature list from checkpoint (version={extra.get('feature_version', '?')})")
            logger.info(f"Loaded trained model from {model_path}")
        else:
            logger.warning(f"No trained model found at {model_path}, using random initialization")
        
        # Initialize dataset builder for training data collection
        dataset_builder = DatasetBuilder()

        # Initialize database (create tables if missing)
        init_db()
        
        # Initialize enroller (storage_dir from config)
        storage_dir = str(config.EMBEDDINGS_DIR)
        state.enroller = DeviceEnroller(
            embedder=state.embedder,
            preprocessor=state.preprocessor,
            feature_converter=state.feature_converter,
            storage_dir=storage_dir,
            dataset_builder=dataset_builder
        )
        
        # Initialize authenticator
        state.authenticator = DeviceAuthenticator(
            embedder=state.embedder,
            preprocessor=state.preprocessor,
            feature_converter=state.feature_converter,
            enroller=state.enroller,
            threshold=0.70
        )
        
        # Initialize challenge-response protocol (DB-backed store)
        challenge_store = DbChallengeStore()
        state.challenge_protocol = ChallengeResponseProtocol(
            nonce_length=32,
            challenge_expiry_seconds=60,
            challenge_store=challenge_store
        )
        
        state.auth_flow = SecureAuthenticationFlow(
            protocol=state.challenge_protocol,
            similarity_threshold=0.85
        )
        
        logger.info("QNA-Auth system initialized successfully")
        _log_local_network_urls(port=8000)
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise


def _log_local_network_urls(port: int = 8000) -> None:
    """Log URLs other devices on the local network can use to reach this server."""
    try:
        hostname = socket.gethostname()
        # Get all IPv4 addresses (skip loopback)
        local_ips = []
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127."):
                local_ips.append(addr)
        # Also try a direct approach in case getaddrinfo only returns loopback
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            direct = s.getsockname()[0]
            if direct not in local_ips:
                local_ips.append(direct)
        for ip in sorted(set(local_ips)):
            logger.info("Local network: other devices can use http://%s:%s", ip, port)
    except Exception:
        pass


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health (includes DB status)"""
    resp = {
        "status": "healthy",
        "model_loaded": state.embedder is not None,
        "components_initialized": all([
            state.embedder,
            state.preprocessor,
            state.enroller,
            state.authenticator,
            state.challenge_protocol
        ]),
        "db_ok": None,
        "devices_in_db": None
    }
    try:
        SessionLocal = get_session_factory()
        session = SessionLocal()
        try:
            count = session.query(Device).count()
            resp["db_ok"] = True
            resp["devices_in_db"] = count
        finally:
            session.close()
    except Exception as e:
        logger.warning("Health check: DB unreachable: %s", e)
        resp["db_ok"] = False
    return resp


# Enrollment endpoint
@app.post("/enroll", response_model=EnrollmentResponse, status_code=status.HTTP_201_CREATED)
async def enroll_device(request: EnrollmentRequest, background_tasks: BackgroundTasks):
    """
    Enroll a new device
    
    This endpoint collects noise samples and creates a device embedding
    """
    if not state.enroller:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enrollment service not initialized"
        )
    
    try:
        logger.info(f"Enrollment request: device_name={request.device_name}, "
                   f"num_samples={request.num_samples}, sources={request.sources}")
        
        # Enroll device
        device_id = state.enroller.enroll_device(
            device_name=request.device_name,
            num_samples=request.num_samples,
            sources=request.sources,
            client_samples=request.client_samples
        )
        
        # Load metadata and persist to DB
        import json
        metadata_path = state.enroller.storage_dir / f"{device_id}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        SessionLocal = get_session_factory()
        session = SessionLocal()
        try:
            session.add(Device(
                device_id=device_id,
                device_name=request.device_name,
                embedding_path=f"{device_id}_embedding.pt",
                metadata_json=json.dumps(metadata)
            ))
            session.commit()
        finally:
            session.close()
        
        return {
            "device_id": device_id,
            "status": "success",
            "message": f"Device enrolled successfully",
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Enrollment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enrollment failed: {str(e)}"
        )


# Authentication endpoint (simple version)
@app.post("/authenticate")
async def authenticate_device(request: AuthenticationRequest):
    """
    Authenticate a device using noise samples
    
    This is a simplified authentication flow
    """
    if not state.authenticator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not initialized"
        )
    
    try:
        logger.info(f"Authentication request: device_id={request.device_id}")
        
        # DEBUG: Check if client_samples arrived
        if request.client_samples:
            logger.info("API: received client_samples in request")
            logger.info(f"API: keys={list(request.client_samples.keys())}")
            for k, v in request.client_samples.items():
                logger.info(f"API: source {k} has {len(v)} samples")
        else:
            logger.warning("API: request.client_samples is None or Empty")

        # Authenticate
        is_authenticated, details = state.authenticator.authenticate(
            device_id=request.device_id,
            sources=request.sources,
            num_samples_per_source=request.num_samples_per_source,
            client_samples=request.client_samples
        )
        
        if is_authenticated:
            # Flatten details into the response to make 'similarity' top-level if needed,
            # BUT VerifyResponse model says 'details' is a Dict. 
            # The test script expects 'similarity' at the top level of the JSON response 
            # because valid VerifyResponse has 'similarity'.
            # However, the code below puts 'similarity' inside 'details' logic?
            # Wait, let's look at VerifyResponse definition:
            # class VerifyResponse(BaseModel):
            #     authenticated: bool
            #     similarity: float
            #     details: Dict
            
            # The return dictionary below must match VerifyResponse fields.
            return {
                "authenticated": True,
                "device_id": request.device_id,
                "similarity": details.get('similarity', 0.0),
                "details": details
            }
        else:
            # Even on failure, we might want to return details if it was a similarity failure
            # But the current logic raises 401. 
            # Note: The test script calls /authenticate (which is this endpoint) and reads the JSON.
            # If we raise 401, requests.post(...).json() might fail or return error details.
            # The test script seems to expect a 200 OK with authenticated=False for rejection test?
            # Let's check the test script.
            # If step 3 is "Attacking", we expect rejection.
            
            # For the purpose of the test script, let's return a clean response 
            # instead of 401 if it's just a similarity mismatch.
            return {
                "authenticated": False,
                "device_id": request.device_id,
                "similarity": details.get('similarity', 0.0),
                "details": details
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication error: {str(e)}"
        )


# Challenge-response endpoints
@app.post("/challenge", response_model=ChallengeResponse)
async def create_challenge(request: ChallengeRequest):
    """
    Create authentication challenge
    
    Step 1 of challenge-response protocol
    """
    if not state.challenge_protocol:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Challenge protocol not initialized"
        )
    
    try:
        challenge = state.challenge_protocol.create_challenge(request.device_id)
        return challenge
        
    except Exception as e:
        logger.error(f"Challenge creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Challenge creation failed: {str(e)}"
        )


@app.post("/verify", response_model=VerifyResponse)
async def verify_challenge_response(request: VerifyRequest):
    """
    Verify challenge response
    
    Step 2 of challenge-response protocol
    """
    if not state.auth_flow or not state.enroller:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification service not initialized"
        )
    
    try:
        # Load stored embedding
        stored_embedding = state.enroller.load_device_embedding(request.device_id)
        
        if stored_embedding is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not enrolled"
            )
        
        # Convert noise samples to embedding
        noise_arrays = [np.array(sample, dtype=np.float32) for sample in request.noise_samples]
        auth_embedding = state.authenticator.generate_authentication_embedding(noise_arrays)
        
        if auth_embedding is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to generate authentication embedding"
            )
        
        # Complete authentication
        is_authenticated, details = state.auth_flow.complete_authentication(
            challenge_id=request.challenge_id,
            response=request.response,
            auth_embedding=auth_embedding,
            stored_embedding=stored_embedding
        )
        
        if is_authenticated:
            return {
                "authenticated": True,
                "similarity": details['embedding_similarity'],
                "details": details
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Verification failed",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification error: {str(e)}"
        )


# Device management endpoints
@app.get("/devices", response_model=DeviceListResponse)
async def list_devices():
    """List all enrolled devices (from DB; fallback to file scan if DB empty)"""
    if not state.enroller:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enrollment service not initialized"
        )
    
    try:
        SessionLocal = get_session_factory()
        session = SessionLocal()
        try:
            rows = session.query(Device).all()
            devices = [{"device_id": r.device_id, "device_name": r.device_name} for r in rows]
        finally:
            session.close()
        if not devices:
            # Fallback: file scan returns IDs only; device_name will be null
            ids = state.enroller.list_enrolled_devices()
            devices = [{"device_id": d, "device_name": None} for d in ids]
        return {
            "devices": devices,
            "count": len(devices)
        }
    except Exception as e:
        logger.error(f"Failed to list devices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list devices: {str(e)}"
        )


@app.get("/devices/{device_id}")
async def get_device_info(device_id: str):
    """Get device metadata (from DB; fallback to file if not in DB)"""
    if not state.enroller:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enrollment service not initialized"
        )
    
    try:
        import json
        SessionLocal = get_session_factory()
        session = SessionLocal()
        try:
            row = session.query(Device).filter(Device.device_id == device_id).first()
            if row and row.metadata_json:
                return json.loads(row.metadata_json)
        finally:
            session.close()
        metadata_path = state.enroller.storage_dir / f"{device_id}_metadata.json"
        if not metadata_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found"
            )
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get device info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get device info: {str(e)}"
        )


@app.delete("/devices/{device_id}")
async def delete_device(device_id: str):
    """Delete enrolled device (from DB and files)"""
    if not state.enroller:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enrollment service not initialized"
        )
    
    try:
        embedding_path = state.enroller.storage_dir / f"{device_id}_embedding.pt"
        metadata_path = state.enroller.storage_dir / f"{device_id}_metadata.json"
        if not embedding_path.exists():
            # Check DB in case file was removed but row remains
            SessionLocal = get_session_factory()
            session = SessionLocal()
            try:
                session.query(Device).filter(Device.device_id == device_id).delete()
                session.commit()
            finally:
                session.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found"
            )
        SessionLocal = get_session_factory()
        session = SessionLocal()
        try:
            session.query(Device).filter(Device.device_id == device_id).delete()
            session.commit()
        finally:
            session.close()
        embedding_path.unlink()
        if metadata_path.exists():
            metadata_path.unlink()
        logger.info(f"Deleted device: {device_id}")
        return {
            "status": "success",
            "message": f"Device {device_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete device: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete device: {str(e)}"
        )


# Root endpoint
@app.get("/")
async def root():
    """API root"""
    return {
        "service": "QNA-Auth API",
        "version": "1.0.0",
        "description": "Quantum Noise Assisted Authentication System",
        "endpoints": {
            "health": "/health",
            "enroll": "/enroll",
            "authenticate": "/authenticate",
            "challenge": "/challenge",
            "verify": "/verify",
            "devices": "/devices"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
