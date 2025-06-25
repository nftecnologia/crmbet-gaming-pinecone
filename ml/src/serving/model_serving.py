"""
Enterprise Model Serving System - High-availability ML deployment
Production-ready model serving with versioning, A/B testing, and canary deployments.

@author: ML Engineering Team
@version: 2.0.0
@domain: Enterprise Model Deployment
@features: TensorFlow Serving, A/B Testing, Canary Deployment, Auto-rollback
"""

import os
import json
import time
import asyncio
import logging
import threading
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import uuid
import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import grpc
import mlflow
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics for model serving
MODEL_REQUESTS = Counter('model_requests_total', 'Total model requests', ['model_name', 'version'])
MODEL_LATENCY = Histogram('model_latency_seconds', 'Model inference latency', ['model_name', 'version'])
MODEL_ERRORS = Counter('model_errors_total', 'Model inference errors', ['model_name', 'version', 'error_type'])
ACTIVE_MODELS = Gauge('active_models_count', 'Number of active models')
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy score', ['model_name', 'version'])
AB_TEST_TRAFFIC = Counter('ab_test_traffic_total', 'A/B test traffic distribution', ['experiment_id', 'variant'])

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    version: str
    model_path: str
    serving_endpoint: str
    max_batch_size: int = 32
    timeout_ms: int = 5000
    health_check_interval: int = 30

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    experiment_id: str
    model_a: ModelConfig
    model_b: ModelConfig
    traffic_split: float = 0.5  # 50/50 split
    success_metric: str = 'accuracy'
    min_samples: int = 1000
    confidence_threshold: float = 0.95

@dataclass
class PredictionRequest:
    """Prediction request"""
    request_id: str
    user_id: str
    features: Dict[str, float]
    model_name: Optional[str] = None
    model_version: Optional[str] = None

@dataclass
class PredictionResponse:
    """Prediction response"""
    request_id: str
    user_id: str
    prediction: Dict[str, Any]
    model_name: str
    model_version: str
    latency_ms: float
    confidence: float
    timestamp: float

class ModelRegistry:
    """MLflow-based model registry with versioning"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.models = {}  # Cache active models
        
    def register_model(self, model_name: str, model_path: str, version: str = None) -> str:
        """Register model in MLflow registry"""
        try:
            # Register model
            model_version = mlflow.register_model(
                model_uri=model_path,
                name=model_name
            )
            
            registered_version = model_version.version if not version else version
            
            # Transition to staging
            self.client.transition_model_version_stage(
                name=model_name,
                version=registered_version,
                stage="Staging"
            )
            
            logger.info(f"Registered model {model_name} version {registered_version}")
            return registered_version
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise
    
    def promote_model(self, model_name: str, version: str):
        """Promote model to production"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            logger.info(f"Promoted model {model_name} version {version} to production")
        except Exception as e:
            logger.error(f"Model promotion failed: {e}")
            raise
    
    def get_model_versions(self, model_name: str) -> List[Dict]:
        """Get all versions of a model"""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    'version': v.version,
                    'stage': v.current_stage,
                    'created_at': v.creation_timestamp,
                    'model_uri': v.source
                }
                for v in versions
            ]
        except Exception as e:
            logger.error(f"Failed to get model versions: {e}")
            return []
    
    def get_production_model(self, model_name: str) -> Optional[str]:
        """Get current production model version"""
        try:
            versions = self.client.get_latest_versions(model_name, stages=["Production"])
            return versions[0].version if versions else None
        except Exception as e:
            logger.error(f"Failed to get production model: {e}")
            return None

class TensorFlowServingClient:
    """TensorFlow Serving client for model inference"""
    
    def __init__(self, serving_host: str = "localhost", serving_port: int = 8500):
        self.serving_host = serving_host
        self.serving_port = serving_port
        self.channel = grpc.insecure_channel(f"{serving_host}:{serving_port}")
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        
    def predict(self, model_name: str, model_version: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction using TensorFlow Serving"""
        try:
            # Create prediction request
            request = predict_pb2.PredictRequest()
            request.model_spec.name = model_name
            request.model_spec.signature_name = 'serving_default'
            
            if model_version:
                request.model_spec.version.value = int(model_version)
            
            # Convert features to tensor
            feature_array = np.array([list(features.values())], dtype=np.float32)
            request.inputs['input'].CopyFrom(
                tf.make_tensor_proto(feature_array, shape=feature_array.shape)
            )
            
            # Make prediction
            start_time = time.time()
            result = self.stub.Predict(request, timeout=5.0)
            latency = (time.time() - start_time) * 1000
            
            # Extract predictions
            predictions = tf.make_ndarray(result.outputs['output'])
            
            return {
                'predictions': predictions.tolist(),
                'latency_ms': latency
            }
            
        except grpc.RpcError as e:
            logger.error(f"TensorFlow Serving error: {e}")
            raise
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

class ABTestManager:
    """A/B testing framework for model experiments"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.active_experiments = {}
        
    def create_experiment(self, config: ABTestConfig) -> str:
        """Create new A/B test experiment"""
        try:
            # Store experiment config
            self.redis.hset(
                f"experiment:{config.experiment_id}",
                mapping={
                    'config': json.dumps(asdict(config)),
                    'status': 'active',
                    'created_at': datetime.now().isoformat(),
                    'samples_a': 0,
                    'samples_b': 0,
                    'success_a': 0,
                    'success_b': 0
                }
            )
            
            self.active_experiments[config.experiment_id] = config
            logger.info(f"Created A/B test experiment {config.experiment_id}")
            
            return config.experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
    
    def route_request(self, experiment_id: str, user_id: str) -> str:
        """Route request to model variant based on A/B test"""
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.active_experiments[experiment_id]
        
        # Use consistent hashing for user assignment
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        variant = 'a' if (user_hash % 100) / 100 < config.traffic_split else 'b'
        
        # Track traffic distribution
        AB_TEST_TRAFFIC.labels(experiment_id=experiment_id, variant=variant).inc()
        
        model_config = config.model_a if variant == 'a' else config.model_b
        return model_config.name, model_config.version
    
    def record_result(self, experiment_id: str, user_id: str, success: bool):
        """Record experiment result"""
        try:
            config = self.active_experiments.get(experiment_id)
            if not config:
                return
            
            # Determine which variant was used
            user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            variant = 'a' if (user_hash % 100) / 100 < config.traffic_split else 'b'
            
            # Update counters
            pipe = self.redis.pipeline()
            pipe.hincrby(f"experiment:{experiment_id}", f"samples_{variant}", 1)
            if success:
                pipe.hincrby(f"experiment:{experiment_id}", f"success_{variant}", 1)
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Failed to record result: {e}")
    
    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get A/B test results with statistical significance"""
        try:
            data = self.redis.hgetall(f"experiment:{experiment_id}")
            if not data:
                return {}
            
            samples_a = int(data.get('samples_a', 0))
            samples_b = int(data.get('samples_b', 0))
            success_a = int(data.get('success_a', 0))
            success_b = int(data.get('success_b', 0))
            
            # Calculate conversion rates
            rate_a = success_a / samples_a if samples_a > 0 else 0
            rate_b = success_b / samples_b if samples_b > 0 else 0
            
            # Statistical significance test (simplified)
            total_samples = samples_a + samples_b
            min_samples_met = total_samples >= self.active_experiments[experiment_id].min_samples
            
            # Determine winner (simplified - should use proper statistical tests)
            winner = 'a' if rate_a > rate_b else 'b'
            confidence = abs(rate_a - rate_b) / max(rate_a, rate_b, 0.01)
            
            return {
                'experiment_id': experiment_id,
                'status': data.get('status', 'unknown'),
                'samples_a': samples_a,
                'samples_b': samples_b,
                'success_rate_a': rate_a,
                'success_rate_b': rate_b,
                'winner': winner,
                'confidence': confidence,
                'min_samples_met': min_samples_met,
                'statistical_significance': confidence > 0.05 and min_samples_met
            }
            
        except Exception as e:
            logger.error(f"Failed to get experiment results: {e}")
            return {}

class ModelHealthMonitor:
    """Monitor model health and performance"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.health_checks = {}
        self.performance_history = {}
        
    def record_prediction(self, model_name: str, version: str, latency_ms: float, 
                         success: bool, prediction_quality: float = None):
        """Record prediction metrics"""
        try:
            timestamp = int(time.time())
            key = f"model_metrics:{model_name}:{version}:{timestamp//60}"  # Per minute
            
            pipe = self.redis.pipeline()
            pipe.hincrby(key, 'total_requests', 1)
            pipe.hincrbyfloat(key, 'total_latency', latency_ms)
            
            if success:
                pipe.hincrby(key, 'successful_requests', 1)
            else:
                pipe.hincrby(key, 'failed_requests', 1)
            
            if prediction_quality is not None:
                pipe.hincrbyfloat(key, 'quality_score', prediction_quality)
            
            pipe.expire(key, 3600)  # Keep for 1 hour
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Failed to record prediction metrics: {e}")
    
    def get_model_health(self, model_name: str, version: str, 
                        window_minutes: int = 5) -> Dict[str, Any]:
        """Get model health metrics"""
        try:
            current_time = int(time.time())
            start_time = current_time - (window_minutes * 60)
            
            total_requests = 0
            total_latency = 0
            successful_requests = 0
            failed_requests = 0
            quality_scores = []
            
            for minute in range(start_time//60, current_time//60 + 1):
                key = f"model_metrics:{model_name}:{version}:{minute}"
                data = self.redis.hgetall(key)
                
                if data:
                    total_requests += int(data.get('total_requests', 0))
                    total_latency += float(data.get('total_latency', 0))
                    successful_requests += int(data.get('successful_requests', 0))
                    failed_requests += int(data.get('failed_requests', 0))
                    
                    if 'quality_score' in data:
                        quality_scores.append(float(data['quality_score']))
            
            # Calculate metrics
            avg_latency = total_latency / total_requests if total_requests > 0 else 0
            success_rate = successful_requests / total_requests if total_requests > 0 else 0
            error_rate = failed_requests / total_requests if total_requests > 0 else 0
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            
            # Health status
            health_status = 'healthy'
            if error_rate > 0.05:  # > 5% error rate
                health_status = 'unhealthy'
            elif avg_latency > 1000:  # > 1s latency
                health_status = 'degraded'
            elif success_rate < 0.95:  # < 95% success rate
                health_status = 'warning'
            
            return {
                'model_name': model_name,
                'version': version,
                'health_status': health_status,
                'total_requests': total_requests,
                'avg_latency_ms': avg_latency,
                'success_rate': success_rate,
                'error_rate': error_rate,
                'avg_quality_score': avg_quality,
                'window_minutes': window_minutes
            }
            
        except Exception as e:
            logger.error(f"Failed to get model health: {e}")
            return {'health_status': 'unknown'}

class ModelServingSystem:
    """
    Enterprise model serving system with advanced deployment capabilities.
    
    Features:
    - TensorFlow Serving integration
    - MLflow model registry
    - A/B testing framework
    - Canary deployments
    - Health monitoring
    - Auto-rollback on degradation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.registry = ModelRegistry(
            self.config.get('mlflow_uri', 'http://localhost:5000')
        )
        
        self.tf_serving = TensorFlowServingClient(
            self.config.get('tf_serving_host', 'localhost'),
            self.config.get('tf_serving_port', 8500)
        )
        
        self.redis = redis.Redis(
            host=self.config.get('redis_host', 'localhost'),
            port=self.config.get('redis_port', 6379),
            decode_responses=True
        )
        
        self.ab_manager = ABTestManager(self.redis)
        self.health_monitor = ModelHealthMonitor(self.redis)
        
        # Active models and experiments
        self.active_models = {}
        self.canary_deployments = {}
        
        # Performance tracking
        self.request_count = 0
        self.start_time = time.time()
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Main prediction endpoint with A/B testing support"""
        start_time = time.time()
        
        try:
            # Determine model version (A/B test or specified)
            model_name = request.model_name or 'default_cluster_model'
            model_version = request.model_version
            
            # Check for active A/B tests
            experiment_id = self._get_active_experiment(model_name)
            if experiment_id and not model_version:
                model_name, model_version = self.ab_manager.route_request(
                    experiment_id, request.user_id
                )
            
            # Get production version if not specified
            if not model_version:
                model_version = self.registry.get_production_model(model_name)
                if not model_version:
                    raise HTTPException(status_code=404, detail="No production model found")
            
            # Make prediction
            result = self.tf_serving.predict(model_name, model_version, request.features)
            
            # Extract prediction and confidence
            predictions = result['predictions'][0]
            cluster_id = int(np.argmax(predictions))
            confidence = float(np.max(predictions))
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create response
            response = PredictionResponse(
                request_id=request.request_id,
                user_id=request.user_id,
                prediction={
                    'cluster_id': cluster_id,
                    'probabilities': predictions.tolist(),
                    'confidence': confidence
                },
                model_name=model_name,
                model_version=model_version,
                latency_ms=latency_ms,
                confidence=confidence,
                timestamp=time.time()
            )
            
            # Record metrics
            MODEL_REQUESTS.labels(model_name=model_name, version=model_version).inc()
            MODEL_LATENCY.labels(model_name=model_name, version=model_version).observe(latency_ms/1000)
            
            # Health monitoring
            self.health_monitor.record_prediction(
                model_name, model_version, latency_ms, True, confidence
            )
            
            self.request_count += 1
            
            return response
            
        except Exception as e:
            # Record error
            MODEL_ERRORS.labels(
                model_name=model_name or 'unknown',
                version=model_version or 'unknown',
                error_type=type(e).__name__
            ).inc()
            
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def _get_active_experiment(self, model_name: str) -> Optional[str]:
        """Get active A/B test experiment for model"""
        # Check Redis for active experiments
        for exp_id in self.ab_manager.active_experiments:
            config = self.ab_manager.active_experiments[exp_id]
            if config.model_a.name == model_name or config.model_b.name == model_name:
                return exp_id
        return None
    
    def deploy_canary(self, model_name: str, new_version: str, traffic_percentage: float = 10.0):
        """Deploy canary version with limited traffic"""
        try:
            canary_config = {
                'model_name': model_name,
                'canary_version': new_version,
                'production_version': self.registry.get_production_model(model_name),
                'traffic_percentage': traffic_percentage,
                'start_time': time.time(),
                'success_threshold': 0.95,
                'latency_threshold': 1000,  # ms
                'min_requests': 100
            }
            
            self.canary_deployments[model_name] = canary_config
            logger.info(f"Started canary deployment for {model_name} v{new_version}")
            
            # Monitor canary performance
            asyncio.create_task(self._monitor_canary(model_name))
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            raise
    
    async def _monitor_canary(self, model_name: str):
        """Monitor canary deployment and auto-rollback if needed"""
        try:
            config = self.canary_deployments[model_name]
            
            while model_name in self.canary_deployments:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Get health metrics for canary
                canary_health = self.health_monitor.get_model_health(
                    model_name, config['canary_version'], window_minutes=5
                )
                
                prod_health = self.health_monitor.get_model_health(
                    model_name, config['production_version'], window_minutes=5
                )
                
                # Check if canary should be rolled back
                should_rollback = (
                    canary_health['error_rate'] > 0.05 or
                    canary_health['avg_latency_ms'] > config['latency_threshold'] or
                    canary_health['success_rate'] < config['success_threshold']
                )
                
                if should_rollback and canary_health['total_requests'] >= config['min_requests']:
                    logger.warning(f"Rolling back canary deployment for {model_name}")
                    self._rollback_canary(model_name)
                    break
                
                # Check if canary should be promoted
                elif (canary_health['total_requests'] >= config['min_requests'] and
                      canary_health['success_rate'] >= config['success_threshold'] and
                      canary_health['avg_latency_ms'] <= config['latency_threshold']):
                    
                    logger.info(f"Promoting canary to production for {model_name}")
                    self._promote_canary(model_name)
                    break
                    
        except Exception as e:
            logger.error(f"Canary monitoring error: {e}")
    
    def _rollback_canary(self, model_name: str):
        """Rollback canary deployment"""
        if model_name in self.canary_deployments:
            del self.canary_deployments[model_name]
            logger.info(f"Canary rollback completed for {model_name}")
    
    def _promote_canary(self, model_name: str):
        """Promote canary to production"""
        if model_name in self.canary_deployments:
            config = self.canary_deployments[model_name]
            
            # Promote in MLflow
            self.registry.promote_model(model_name, config['canary_version'])
            
            # Remove canary config
            del self.canary_deployments[model_name]
            
            logger.info(f"Canary promoted to production for {model_name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = time.time() - self.start_time
        avg_throughput = self.request_count / uptime if uptime > 0 else 0
        
        return {
            'status': 'healthy',
            'uptime_seconds': uptime,
            'total_requests': self.request_count,
            'avg_throughput_rps': avg_throughput,
            'active_models': len(self.active_models),
            'active_experiments': len(self.ab_manager.active_experiments),
            'canary_deployments': len(self.canary_deployments),
            'memory_usage_gb': psutil.virtual_memory().used / (1024**3),
            'cpu_percent': psutil.cpu_percent()
        }


# FastAPI application for model serving
app = FastAPI(title="Enterprise ML Model Serving", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global serving system instance
serving_system = ModelServingSystem()

# Pydantic models for API
class PredictionRequestAPI(BaseModel):
    user_id: str
    features: Dict[str, float]
    model_name: Optional[str] = None
    model_version: Optional[str] = None

@app.post("/predict")
async def predict_endpoint(request: PredictionRequestAPI):
    """Main prediction endpoint"""
    prediction_request = PredictionRequest(
        request_id=str(uuid.uuid4()),
        user_id=request.user_id,
        features=request.features,
        model_name=request.model_name,
        model_version=request.model_version
    )
    
    response = await serving_system.predict(prediction_request)
    return response.to_dict() if hasattr(response, 'to_dict') else response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return serving_system.get_system_status()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/deploy/canary")
async def deploy_canary(model_name: str, version: str, traffic_percentage: float = 10.0):
    """Deploy canary version"""
    serving_system.deploy_canary(model_name, version, traffic_percentage)
    return {"status": "canary deployment started"}

@app.get("/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get A/B test results"""
    return serving_system.ab_manager.get_experiment_results(experiment_id)


# Example usage and testing
if __name__ == "__main__":
    logger.info("🚀 Starting Enterprise Model Serving System")
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        workers=4,
        loop="asyncio",
        access_log=True
    )