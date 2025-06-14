Production Deployment Guide
===========================

This comprehensive guide covers deploying the OEE Forecasting and Analytics system in production environments, including scalability considerations, security measures, monitoring setup, and best practices for enterprise deployment.

🚀 **Deployment Architecture Overview**
======================================

**Production Deployment Architecture:**

.. code-block::

   Production Architecture:
   
   ┌─────────────────────────────────────────────────────────────┐
   │                    LOAD BALANCER                            │
   │                 (nginx/HAProxy)                             │
   └─────────────────────────────────────────────────────────────┘
                                │
   ┌─────────────────────────────────────────────────────────────┐
   │                 APPLICATION LAYER                           │
   │  ┌─────────────────┐  ┌─────────────────┐                  │
   │  │ Streamlit App   │  │ API Gateway     │                  │
   │  │ Instance 1      │  │ (FastAPI)       │                  │
   │  └─────────────────┘  └─────────────────┘                  │
   │  ┌─────────────────┐  ┌─────────────────┐                  │
   │  │ Streamlit App   │  │ Background      │                  │
   │  │ Instance 2      │  │ Workers         │                  │
   │  └─────────────────┘  └─────────────────┘                  │
   └─────────────────────────────────────────────────────────────┘
                                │
   ┌─────────────────────────────────────────────────────────────┐
   │                  PROCESSING LAYER                          │
   │  ┌─────────────────┐  ┌─────────────────┐                  │
   │  │ Model Serving   │  │ RAG System      │                  │
   │  │ (TensorFlow     │  │ (Embeddings +   │                  │
   │  │  Serving)       │  │  Vector DB)     │                  │
   │  └─────────────────┘  └─────────────────┘                  │
   │  ┌─────────────────┐  ┌─────────────────┐                  │
   │  │ Data Processing │  │ Cache Layer     │                  │
   │  │ Pipeline        │  │ (Redis)         │                  │
   │  └─────────────────┘  └─────────────────┘                  │
   └─────────────────────────────────────────────────────────────┘
                                │
   ┌─────────────────────────────────────────────────────────────┐
   │                    STORAGE LAYER                           │
   │  ┌─────────────────┐  ┌─────────────────┐                  │
   │  │ PostgreSQL      │  │ File Storage    │                  │
   │  │ (Metadata)      │  │ (S3/MinIO)      │                  │
   │  └─────────────────┘  └─────────────────┘                  │
   │  ┌─────────────────┐  ┌─────────────────┐                  │
   │  │ Vector Database │  │ Model Registry  │                  │
   │  │ (FAISS/Weaviate)│  │ (MLflow)        │                  │
   │  └─────────────────┘  └─────────────────┘                  │
   └─────────────────────────────────────────────────────────────┘

**Deployment Strategies:**

.. code-block::

   Deployment Options:
   
   1. Single Server Deployment
      ├── Suitable for: Small to medium manufacturing facilities
      ├── Components: All services on one server
      ├── Scaling: Vertical scaling only
      └── Complexity: Low
   
   2. Multi-Server Deployment
      ├── Suitable for: Large manufacturing facilities
      ├── Components: Distributed across multiple servers
      ├── Scaling: Horizontal and vertical scaling
      └── Complexity: Medium
   
   3. Container-based Deployment (Docker/Kubernetes)
      ├── Suitable for: Enterprise environments
      ├── Components: Containerized microservices
      ├── Scaling: Auto-scaling capabilities
      └── Complexity: High
   
   4. Cloud-native Deployment
      ├── Suitable for: Multi-facility enterprises
      ├── Components: Cloud-managed services
      ├── Scaling: Elastic scaling
      └── Complexity: Medium-High

🐳 **Containerized Deployment**
==============================

**Docker Configuration**

.. code-block:: dockerfile

   # Dockerfile for OEE Analytics Application
   FROM python:3.9-slim

   # Set working directory
   WORKDIR /app

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       gcc \
       g++ \
       curl \
       && rm -rf /var/lib/apt/lists/*

   # Copy requirements and install Python dependencies
   COPY requirements.txt .
   COPY requirements_rag.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   RUN pip install --no-cache-dir -r requirements_rag.txt

   # Copy application code
   COPY . .

   # Create non-root user
   RUN useradd -m -u 1000 oeeuser && chown -R oeeuser:oeeuser /app
   USER oeeuser

   # Expose port
   EXPOSE 8501

   # Health check
   HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
     CMD curl -f http://localhost:8501/_stcore/health || exit 1

   # Start application
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

**Docker Compose for Multi-Service Setup**

.. code-block:: yaml

   # docker-compose.yml
   version: '3.8'

   services:
     # Main Streamlit Application
     oee-app:
       build: .
       ports:
         - "8501:8501"
       environment:
         - GEMINI_API_KEY=${GEMINI_API_KEY}
         - POSTGRES_URL=postgresql://postgres:password@postgres:5432/oee_db
         - REDIS_URL=redis://redis:6379
       volumes:
         - ./data:/app/data
         - ./models:/app/models
       depends_on:
         - postgres
         - redis
       restart: unless-stopped

     # API Gateway
     api-gateway:
       build:
         context: .
         dockerfile: Dockerfile.api
       ports:
         - "8000:8000"
       environment:
         - DATABASE_URL=postgresql://postgres:password@postgres:5432/oee_db
         - REDIS_URL=redis://redis:6379
       depends_on:
         - postgres
         - redis
       restart: unless-stopped

     # PostgreSQL Database
     postgres:
       image: postgres:13
       environment:
         POSTGRES_DB: oee_db
         POSTGRES_USER: postgres
         POSTGRES_PASSWORD: password
       volumes:
         - postgres_data:/var/lib/postgresql/data
       ports:
         - "5432:5432"
       restart: unless-stopped

     # Redis Cache
     redis:
       image: redis:6-alpine
       ports:
         - "6379:6379"
       volumes:
         - redis_data:/data
       restart: unless-stopped

     # Model Serving
     model-server:
       build:
         context: .
         dockerfile: Dockerfile.model-server
       ports:
         - "8501:8501"
       volumes:
         - ./models:/models
       environment:
         - MODEL_PATH=/models
       restart: unless-stopped

     # Background Workers
     worker:
       build: .
       command: python -m celery worker -A tasks.celery --loglevel=info
       environment:
         - REDIS_URL=redis://redis:6379
         - DATABASE_URL=postgresql://postgres:password@postgres:5432/oee_db
       depends_on:
         - postgres
         - redis
       restart: unless-stopped

     # Load Balancer
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./ssl:/etc/nginx/ssl
       depends_on:
         - oee-app
         - api-gateway
       restart: unless-stopped

   volumes:
     postgres_data:
     redis_data:

**Kubernetes Deployment**

.. code-block:: yaml

   # k8s-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: oee-analytics
     labels:
       app: oee-analytics
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: oee-analytics
     template:
       metadata:
         labels:
           app: oee-analytics
       spec:
         containers:
         - name: oee-app
           image: oee-analytics:latest
           ports:
           - containerPort: 8501
           env:
           - name: GEMINI_API_KEY
             valueFrom:
               secretKeyRef:
                 name: oee-secrets
                 key: gemini-api-key
           - name: DATABASE_URL
             valueFrom:
               configMapKeyRef:
                 name: oee-config
                 key: database-url
           resources:
             requests:
               memory: "1Gi"
               cpu: "500m"
             limits:
               memory: "2Gi"
               cpu: "1000m"
           livenessProbe:
             httpGet:
               path: /_stcore/health
               port: 8501
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /_stcore/health
               port: 8501
             initialDelaySeconds: 5
             periodSeconds: 5

   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: oee-analytics-service
   spec:
     selector:
       app: oee-analytics
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8501
     type: LoadBalancer

🔧 **Infrastructure Configuration**
==================================

**Load Balancer Configuration (Nginx)**

.. code-block:: nginx

   # nginx.conf
   events {
       worker_connections 1024;
   }

   http {
       upstream oee_app {
           server oee-app:8501;
       }

       upstream api_gateway {
           server api-gateway:8000;
       }

       # Rate limiting
       limit_req_zone $binary_remote_addr zone=app_limit:10m rate=10r/s;
       limit_req_zone $binary_remote_addr zone=api_limit:10m rate=50r/s;

       server {
           listen 80;
           server_name your-domain.com;

           # Redirect HTTP to HTTPS
           return 301 https://$server_name$request_uri;
       }

       server {
           listen 443 ssl http2;
           server_name your-domain.com;

           ssl_certificate /etc/nginx/ssl/cert.pem;
           ssl_certificate_key /etc/nginx/ssl/key.pem;

           # Security headers
           add_header X-Frame-Options DENY;
           add_header X-Content-Type-Options nosniff;
           add_header X-XSS-Protection "1; mode=block";
           add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

           # Main application
           location / {
               limit_req zone=app_limit burst=20 nodelay;
               proxy_pass http://oee_app;
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
               proxy_set_header X-Forwarded-Proto $scheme;
               
               # WebSocket support for Streamlit
               proxy_http_version 1.1;
               proxy_set_header Upgrade $http_upgrade;
               proxy_set_header Connection "upgrade";
           }

           # API endpoints
           location /api/ {
               limit_req zone=api_limit burst=100 nodelay;
               proxy_pass http://api_gateway;
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
               proxy_set_header X-Forwarded-Proto $scheme;
           }

           # Health check endpoint
           location /health {
               access_log off;
               return 200 "healthy\n";
               add_header Content-Type text/plain;
           }
       }
   }

**Database Configuration**

.. code-block:: sql

   -- PostgreSQL setup for OEE Analytics
   
   -- Create database
   CREATE DATABASE oee_analytics;
   
   -- Create user with limited privileges
   CREATE USER oee_user WITH PASSWORD 'secure_password';
   GRANT CONNECT ON DATABASE oee_analytics TO oee_user;
   
   -- Use the database
   \c oee_analytics;
   
   -- Create tables for metadata storage
   CREATE TABLE model_metadata (
       id SERIAL PRIMARY KEY,
       model_name VARCHAR(255) NOT NULL,
       model_version VARCHAR(50) NOT NULL,
       production_line VARCHAR(50),
       training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       performance_metrics JSONB,
       model_path TEXT,
       is_active BOOLEAN DEFAULT false,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   
   CREATE TABLE prediction_logs (
       id SERIAL PRIMARY KEY,
       model_name VARCHAR(255),
       production_line VARCHAR(50),
       prediction_date DATE,
       predicted_oee FLOAT,
       actual_oee FLOAT,
       prediction_confidence FLOAT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   
   CREATE TABLE system_metrics (
       id SERIAL PRIMARY KEY,
       metric_name VARCHAR(255),
       metric_value FLOAT,
       metadata JSONB,
       recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   
   -- Grant permissions
   GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO oee_user;
   GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO oee_user;

🔒 **Security Configuration**
============================

**Environment Variables and Secrets Management**

.. code-block:: bash

   # .env.production
   # Database Configuration
   DATABASE_URL=postgresql://oee_user:secure_password@postgres:5432/oee_analytics
   
   # Redis Configuration
   REDIS_URL=redis://redis:6379
   
   # API Keys (use secrets management in production)
   GEMINI_API_KEY=your_secure_api_key
   
   # Security Configuration
   SECRET_KEY=your_secret_key_for_sessions
   ALLOWED_HOSTS=your-domain.com,localhost
   
   # Logging Configuration
   LOG_LEVEL=INFO
   LOG_FILE=/app/logs/app.log
   
   # Model Configuration
   MODEL_CACHE_DIR=/app/models
   DOCUMENT_CACHE_DIR=/app/documents

**SSL/TLS Configuration**

.. code-block:: bash

   #!/bin/bash
   # ssl-setup.sh - SSL certificate setup script
   
   # Create SSL directory
   mkdir -p /etc/nginx/ssl
   
   # Generate self-signed certificate for development
   # For production, use Let's Encrypt or purchased certificates
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
       -keyout /etc/nginx/ssl/key.pem \
       -out /etc/nginx/ssl/cert.pem \
       -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=your-domain.com"
   
   # Set proper permissions
   chmod 600 /etc/nginx/ssl/key.pem
   chmod 644 /etc/nginx/ssl/cert.pem
   
   echo "SSL certificates generated successfully"

**Authentication and Authorization**

.. code-block:: python

   # auth.py - Authentication module for production deployment
   
   import jwt
   import bcrypt
   from datetime import datetime, timedelta
   from functools import wraps
   import streamlit as st
   
   class AuthenticationManager:
       def __init__(self, secret_key, token_expiry_hours=24):
           self.secret_key = secret_key
           self.token_expiry = timedelta(hours=token_expiry_hours)
       
       def hash_password(self, password):
           """Hash password using bcrypt"""
           return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
       
       def verify_password(self, password, hashed):
           """Verify password against hash"""
           return bcrypt.checkpw(password.encode('utf-8'), hashed)
       
       def generate_token(self, user_id, permissions):
           """Generate JWT token for authenticated user"""
           payload = {
               'user_id': user_id,
               'permissions': permissions,
               'exp': datetime.utcnow() + self.token_expiry,
               'iat': datetime.utcnow()
           }
           return jwt.encode(payload, self.secret_key, algorithm='HS256')
       
       def verify_token(self, token):
           """Verify and decode JWT token"""
           try:
               payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
               return payload
           except jwt.ExpiredSignatureError:
               return None
           except jwt.InvalidTokenError:
               return None
   
   def require_authentication(permissions=None):
       """Decorator for requiring authentication"""
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               # Check if user is authenticated
               if 'authenticated' not in st.session_state:
                   st.error("Please log in to access this feature")
                   return None
               
               # Check permissions if specified
               if permissions:
                   user_permissions = st.session_state.get('permissions', [])
                   if not any(perm in user_permissions for perm in permissions):
                       st.error("Insufficient permissions")
                       return None
               
               return func(*args, **kwargs)
           return wrapper
       return decorator

📊 **Monitoring and Observability**
===================================

**Application Monitoring Setup**

.. code-block:: python

   # monitoring.py - Comprehensive monitoring system
   
   import prometheus_client
   from prometheus_client import Counter, Histogram, Gauge, start_http_server
   import logging
   import time
   from functools import wraps
   
   # Prometheus metrics
   REQUEST_COUNT = Counter('app_requests_total', 'Total app requests', ['method', 'endpoint'])
   REQUEST_DURATION = Histogram('app_request_duration_seconds', 'Request duration')
   ACTIVE_USERS = Gauge('app_active_users', 'Number of active users')
   MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total model predictions', ['model_type', 'production_line'])
   PREDICTION_ACCURACY = Gauge('model_prediction_accuracy', 'Model prediction accuracy', ['model_type', 'production_line'])
   
   class ApplicationMonitor:
       def __init__(self, metrics_port=9090):
           self.metrics_port = metrics_port
           self.logger = self._setup_logging()
           
       def _setup_logging(self):
           """Setup structured logging"""
           logging.basicConfig(
               level=logging.INFO,
               format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
               handlers=[
                   logging.FileHandler('/app/logs/app.log'),
                   logging.StreamHandler()
               ]
           )
           return logging.getLogger(__name__)
       
       def start_metrics_server(self):
           """Start Prometheus metrics server"""
           start_http_server(self.metrics_port)
           self.logger.info(f"Metrics server started on port {self.metrics_port}")
       
       def track_request(self, method, endpoint):
           """Track request metrics"""
           def decorator(func):
               @wraps(func)
               def wrapper(*args, **kwargs):
                   start_time = time.time()
                   
                   try:
                       result = func(*args, **kwargs)
                       REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
                       return result
                   except Exception as e:
                       self.logger.error(f"Error in {endpoint}: {str(e)}")
                       raise
                   finally:
                       REQUEST_DURATION.observe(time.time() - start_time)
               
               return wrapper
           return decorator
       
       def track_prediction(self, model_type, production_line, accuracy=None):
           """Track model prediction metrics"""
           MODEL_PREDICTIONS.labels(
               model_type=model_type, 
               production_line=production_line
           ).inc()
           
           if accuracy is not None:
               PREDICTION_ACCURACY.labels(
                   model_type=model_type,
                   production_line=production_line
               ).set(accuracy)

**Health Check Implementation**

.. code-block:: python

   # health_check.py - Comprehensive health checking
   
   import asyncio
   import psutil
   import requests
   from datetime import datetime
   import redis
   import psycopg2
   
   class HealthChecker:
       def __init__(self, config):
           self.config = config
           self.checks = {
               'database': self.check_database,
               'redis': self.check_redis,
               'memory': self.check_memory,
               'disk': self.check_disk,
               'models': self.check_models,
               'external_apis': self.check_external_apis
           }
       
       async def run_all_checks(self):
           """Run all health checks asynchronously"""
           results = {}
           
           for check_name, check_func in self.checks.items():
               try:
                   results[check_name] = await asyncio.create_task(check_func())
               except Exception as e:
                   results[check_name] = {
                       'status': 'unhealthy',
                       'error': str(e),
                       'timestamp': datetime.utcnow().isoformat()
                   }
           
           # Overall health status
           overall_status = 'healthy' if all(
               result['status'] == 'healthy' for result in results.values()
           ) else 'unhealthy'
           
           return {
               'overall_status': overall_status,
               'checks': results,
               'timestamp': datetime.utcnow().isoformat()
           }
       
       async def check_database(self):
           """Check database connectivity and performance"""
           try:
               conn = psycopg2.connect(self.config['DATABASE_URL'])
               cursor = conn.cursor()
               cursor.execute('SELECT 1')
               cursor.close()
               conn.close()
               
               return {
                   'status': 'healthy',
                   'message': 'Database connection successful'
               }
           except Exception as e:
               return {
                   'status': 'unhealthy',
                   'error': str(e)
               }
       
       async def check_redis(self):
           """Check Redis connectivity"""
           try:
               r = redis.from_url(self.config['REDIS_URL'])
               r.ping()
               
               return {
                   'status': 'healthy',
                   'message': 'Redis connection successful'
               }
           except Exception as e:
               return {
                   'status': 'unhealthy',
                   'error': str(e)
               }
       
       async def check_memory(self):
           """Check system memory usage"""
           memory = psutil.virtual_memory()
           
           if memory.percent > 90:
               status = 'unhealthy'
               message = f'High memory usage: {memory.percent}%'
           elif memory.percent > 80:
               status = 'warning'
               message = f'Memory usage: {memory.percent}%'
           else:
               status = 'healthy'
               message = f'Memory usage: {memory.percent}%'
           
           return {
               'status': status,
               'message': message,
               'usage_percent': memory.percent
           }

**Grafana Dashboard Configuration**

.. code-block:: json

   {
     "dashboard": {
       "title": "OEE Analytics System Monitoring",
       "panels": [
         {
           "title": "Request Rate",
           "type": "graph",
           "targets": [
             {
               "expr": "rate(app_requests_total[5m])",
               "legendFormat": "Requests/sec"
             }
           ]
         },
         {
           "title": "Response Time",
           "type": "graph",
           "targets": [
             {
               "expr": "histogram_quantile(0.95, rate(app_request_duration_seconds_bucket[5m]))",
               "legendFormat": "95th percentile"
             }
           ]
         },
         {
           "title": "Model Prediction Accuracy",
           "type": "stat",
           "targets": [
             {
               "expr": "model_prediction_accuracy",
               "legendFormat": "{{model_type}} - {{production_line}}"
             }
           ]
         },
         {
           "title": "System Resources",
           "type": "graph",
           "targets": [
             {
               "expr": "cpu_usage_percent",
               "legendFormat": "CPU Usage"
             },
             {
               "expr": "memory_usage_percent",
               "legendFormat": "Memory Usage"
             }
           ]
         }
       ]
     }
   }

🔄 **CI/CD Pipeline**
====================

**GitHub Actions Workflow**

.. code-block:: yaml

   # .github/workflows/deploy.yml
   name: Deploy OEE Analytics

   on:
     push:
       branches: [ main ]
     pull_request:
       branches: [ main ]

   jobs:
     test:
       runs-on: ubuntu-latest
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Set up Python
         uses: actions/setup-python@v4
         with:
           python-version: '3.9'
       
       - name: Install dependencies
         run: |
           python -m pip install --upgrade pip
           pip install -r requirements.txt
           pip install -r requirements_rag.txt
           pip install pytest pytest-cov
       
       - name: Run tests
         run: |
           pytest tests/ --cov=./ --cov-report=xml
       
       - name: Upload coverage to Codecov
         uses: codecov/codecov-action@v3

     build:
       needs: test
       runs-on: ubuntu-latest
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Build Docker image
         run: |
           docker build -t oee-analytics:${{ github.sha }} .
       
       - name: Run security scan
         run: |
           docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
             aquasec/trivy image oee-analytics:${{ github.sha }}

     deploy:
       needs: [test, build]
       runs-on: ubuntu-latest
       if: github.ref == 'refs/heads/main'
       
       steps:
       - uses: actions/checkout@v3
       
       - name: Deploy to staging
         run: |
           # Deploy to staging environment
           echo "Deploying to staging..."
       
       - name: Run integration tests
         run: |
           # Run integration tests against staging
           echo "Running integration tests..."
       
       - name: Deploy to production
         if: success()
         run: |
           # Deploy to production environment
           echo "Deploying to production..."

**Automated Backup Strategy**

.. code-block:: bash

   #!/bin/bash
   # backup.sh - Automated backup script
   
   set -e
   
   # Configuration
   BACKUP_DIR="/backups"
   DATE=$(date +%Y%m%d_%H%M%S)
   RETENTION_DAYS=30
   
   # Database backup
   echo "Starting database backup..."
   pg_dump $DATABASE_URL > "$BACKUP_DIR/database_$DATE.sql"
   
   # Model files backup
   echo "Backing up model files..."
   tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" /app/models/
   
   # Configuration backup
   echo "Backing up configuration..."
   tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" /app/config/
   
   # Upload to cloud storage (S3)
   if [ ! -z "$AWS_S3_BUCKET" ]; then
       echo "Uploading backups to S3..."
       aws s3 cp "$BACKUP_DIR/database_$DATE.sql" "s3://$AWS_S3_BUCKET/backups/"
       aws s3 cp "$BACKUP_DIR/models_$DATE.tar.gz" "s3://$AWS_S3_BUCKET/backups/"
       aws s3 cp "$BACKUP_DIR/config_$DATE.tar.gz" "s3://$AWS_S3_BUCKET/backups/"
   fi
   
   # Cleanup old backups
   echo "Cleaning up old backups..."
   find $BACKUP_DIR -name "*.sql" -mtime +$RETENTION_DAYS -delete
   find $BACKUP_DIR -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
   
   echo "Backup completed successfully"

📈 **Scaling and Performance**
=============================

**Auto-scaling Configuration**

.. code-block:: yaml

   # k8s-autoscaling.yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: oee-analytics-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: oee-analytics
     minReplicas: 2
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
     - type: Resource
       resource:
         name: memory
         target:
           type: Utilization
           averageUtilization: 80
     behavior:
       scaleUp:
         stabilizationWindowSeconds: 60
         policies:
         - type: Percent
           value: 100
           periodSeconds: 15
       scaleDown:
         stabilizationWindowSeconds: 300
         policies:
         - type: Percent
           value: 10
           periodSeconds: 60

**Caching Strategy**

.. code-block:: python

   # caching.py - Multi-level caching implementation
   
   import redis
   import pickle
   import hashlib
   from functools import wraps
   from typing import Any, Optional
   
   class CacheManager:
       def __init__(self, redis_url: str, default_ttl: int = 3600):
           self.redis_client = redis.from_url(redis_url)
           self.default_ttl = default_ttl
       
       def cache_result(self, ttl: Optional[int] = None, key_prefix: str = ""):
           """Decorator for caching function results"""
           def decorator(func):
               @wraps(func)
               def wrapper(*args, **kwargs):
                   # Generate cache key
                   key_data = f"{key_prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
                   cache_key = hashlib.md5(key_data.encode()).hexdigest()
                   
                   # Try to get from cache
                   cached_result = self.get(cache_key)
                   if cached_result is not None:
                       return cached_result
                   
                   # Execute function and cache result
                   result = func(*args, **kwargs)
                   self.set(cache_key, result, ttl or self.default_ttl)
                   return result
               
               return wrapper
           return decorator
       
       def get(self, key: str) -> Any:
           """Get value from cache"""
           try:
               cached_data = self.redis_client.get(key)
               if cached_data:
                   return pickle.loads(cached_data)
           except Exception:
               pass
           return None
       
       def set(self, key: str, value: Any, ttl: int) -> None:
           """Set value in cache"""
           try:
               serialized_data = pickle.dumps(value)
               self.redis_client.setex(key, ttl, serialized_data)
           except Exception:
               pass  # Fail silently for caching errors

🛠️ **Deployment Scripts**
=========================

**One-Click Deployment Script**

.. code-block:: bash

   #!/bin/bash
   # deploy.sh - One-click deployment script
   
   set -e
   
   echo "🚀 Starting OEE Analytics Deployment"
   
   # Check prerequisites
   command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }
   command -v docker-compose >/dev/null 2>&1 || { echo "Docker Compose is required but not installed. Aborting." >&2; exit 1; }
   
   # Configuration
   read -p "Enter your domain name (e.g., oee.yourcompany.com): " DOMAIN_NAME
   read -s -p "Enter your Gemini API key: " GEMINI_API_KEY
   echo
   
   # Create directories
   mkdir -p data logs ssl
   
   # Generate SSL certificates
   if [ ! -f "ssl/cert.pem" ]; then
       echo "📜 Generating SSL certificates..."
       openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
           -keyout ssl/key.pem -out ssl/cert.pem \
           -subj "/C=US/ST=State/L=City/O=Company/CN=$DOMAIN_NAME"
   fi
   
   # Create environment file
   cat > .env.production << EOF
   DOMAIN_NAME=$DOMAIN_NAME
   GEMINI_API_KEY=$GEMINI_API_KEY
   DATABASE_URL=postgresql://postgres:secure_password@postgres:5432/oee_analytics
   REDIS_URL=redis://redis:6379
   SECRET_KEY=$(openssl rand -base64 32)
   EOF
   
   # Build and start services
   echo "🔨 Building and starting services..."
   docker-compose -f docker-compose.production.yml --env-file .env.production up -d --build
   
   # Wait for services to start
   echo "⏳ Waiting for services to start..."
   sleep 30
   
   # Run database migrations
   echo "📊 Setting up database..."
   docker-compose -f docker-compose.production.yml exec -T postgres psql -U postgres -c "CREATE DATABASE IF NOT EXISTS oee_analytics;"
   
   # Health check
   echo "🏥 Performing health check..."
   if curl -f http://localhost/health > /dev/null 2>&1; then
       echo "✅ Deployment successful!"
       echo "📍 Access your application at: https://$DOMAIN_NAME"
       echo "📈 Monitoring dashboard: https://$DOMAIN_NAME/metrics"
   else
       echo "❌ Deployment failed - check logs with: docker-compose logs"
       exit 1
   fi

**Update and Maintenance Script**

.. code-block:: bash

   #!/bin/bash
   # maintenance.sh - System maintenance script
   
   # Backup before update
   echo "📦 Creating backup..."
   ./backup.sh
   
   # Pull latest images
   echo "📥 Pulling latest images..."
   docker-compose pull
   
   # Update application
   echo "🔄 Updating application..."
   docker-compose up -d --force-recreate
   
   # Clean up old images
   echo "🧹 Cleaning up..."
   docker image prune -f
   
   # Verify deployment
   echo "✅ Verifying deployment..."
   ./health_check.sh
   
   echo "🎉 Maintenance completed successfully!"

📚 **Best Practices Summary**
============================

**Security Best Practices:**

.. code-block::

   Security Checklist:
   
   ✅ Use HTTPS for all communications
   ✅ Implement proper authentication and authorization
   ✅ Store secrets in environment variables or secret managers
   ✅ Regular security updates and vulnerability scanning
   ✅ Network segmentation and firewall rules
   ✅ Input validation and sanitization
   ✅ Rate limiting and DDoS protection
   ✅ Regular security audits and penetration testing

**Performance Best Practices:**

.. code-block::

   Performance Optimization:
   
   ✅ Implement multi-level caching
   ✅ Use connection pooling for databases
   ✅ Optimize model serving with batch prediction
   ✅ Use CDN for static assets
   ✅ Implement proper logging and monitoring
   ✅ Regular performance testing and optimization
   ✅ Auto-scaling based on metrics
   ✅ Database query optimization

**Monitoring Best Practices:**

.. code-block::

   Monitoring Strategy:
   
   ✅ Application Performance Monitoring (APM)
   ✅ Infrastructure monitoring (CPU, memory, disk)
   ✅ Business metrics monitoring (OEE accuracy, user satisfaction)
   ✅ Log aggregation and analysis
   ✅ Alerting for critical issues
   ✅ Regular health checks
   ✅ Capacity planning and forecasting
   ✅ Incident response procedures

**Next Steps:**

- Review :doc:`../troubleshooting` for common deployment issues
- Check the monitoring dashboard setup guide
- Implement automated testing for your specific environment
- Setup disaster recovery procedures