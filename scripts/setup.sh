#!/bin/bash

# Carbon Footprint Tracker - Complete Setup Script
# This script sets up the entire development and production environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="carbon-tracker"
ENVIRONMENT=${ENVIRONMENT:-"dev"}
REGION=${AWS_REGION:-"us-west-2"}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if required tools are installed
    local tools=("docker" "docker-compose" "kubectl" "terraform" "aws" "helm")
    local missing_tools=()
    
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and run the script again."
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker and try again."
        exit 1
    fi
    
    log_success "All prerequisites are met!"
}

setup_local_environment() {
    log_info "Setting up local development environment..."
    
    # Create necessary directories
    mkdir -p {data,logs,models,config/{grafana,prometheus,nginx}}
    
    # Create environment file
    cat > .env << EOF
# Carbon Tracker Environment Configuration
ENVIRONMENT=${ENVIRONMENT}
PROJECT_NAME=${PROJECT_NAME}

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=carbon_tracker
POSTGRES_USER=carbon_user
POSTGRES_PASSWORD=carbon_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# InfluxDB Configuration
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=carbon-tracker-super-secret-token
INFLUXDB_ORG=carbon-tracker
INFLUXDB_BUCKET=carbon-emissions

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# ML Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_PATH=./models/carbon_prediction_model.pkl

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=carbontracker123
EOF
    
    log_success "Local environment configuration created!"
}

setup_docker_environment() {
    log_info "Setting up Docker environment..."
    
    # Create Dockerfiles
    create_dockerfiles
    
    # Create docker-compose override for development
    cat > docker-compose.override.yml << EOF
version: '3.8'

services:
  carbon-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    volumes:
      - ./src:/app/src:ro
      - ./models:/app/models:ro
    environment:
      - DEBUG=true
      - RELOAD=true
    ports:
      - "8000:8000"

  carbon-producer:
    build:
      context: .
      dockerfile: Dockerfile.producer
    volumes:
      - ./src:/app/src:ro
    environment:
      - DEBUG=true

  carbon-processor:
    build:
      context: .
      dockerfile: Dockerfile.processor
    volumes:
      - ./src:/app/src:ro
      - ./models:/app/models:ro
    environment:
      - DEBUG=true
EOF
    
    log_success "Docker environment configured!"
}

create_dockerfiles() {
    log_info "Creating Dockerfiles..."
    
    # API Dockerfile
    cat > Dockerfile.api << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.carbon_api:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    # Producer Dockerfile
    cat > Dockerfile.producer << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Run producer
CMD ["python", "src/data_ingestion/kafka_producer.py"]
EOF

    # Processor Dockerfile
    cat > Dockerfile.processor << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Run processor
CMD ["python", "src/stream_processing/flink_processor.py"]
EOF

    # ML Training Dockerfile
    cat > Dockerfile.ml << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Run ML training
CMD ["python", "src/ml_pipeline/carbon_prediction_model.py"]
EOF

    log_success "Dockerfiles created!"
}

setup_kubernetes_configs() {
    log_info "Setting up Kubernetes configurations..."
    
    mkdir -p k8s/{base,overlays/{dev,staging,prod}}
    
    # Create base Kubernetes manifests
    create_k8s_manifests
    
    log_success "Kubernetes configurations created!"
}

create_k8s_manifests() {
    # Namespace
    cat > k8s/base/namespace.yaml << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: carbon-tracker
  labels:
    name: carbon-tracker
EOF

    # ConfigMap
    cat > k8s/base/configmap.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: carbon-tracker-config
  namespace: carbon-tracker
data:
  ENVIRONMENT: "${ENVIRONMENT}"
  KAFKA_BOOTSTRAP_SERVERS: "kafka:9092"
  INFLUXDB_URL: "http://influxdb:8086"
  REDIS_URL: "redis://redis:6379"
EOF

    # API Deployment
    cat > k8s/base/api-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: carbon-api
  namespace: carbon-tracker
spec:
  replicas: 3
  selector:
    matchLabels:
      app: carbon-api
  template:
    metadata:
      labels:
        app: carbon-api
    spec:
      containers:
      - name: carbon-api
        image: carbon-tracker/api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: carbon-tracker-config
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: carbon-api-service
  namespace: carbon-tracker
spec:
  selector:
    app: carbon-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
EOF

    # Ingress
    cat > k8s/base/ingress.yaml << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: carbon-tracker-ingress
  namespace: carbon-tracker
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.carbon-tracker.com
    secretName: carbon-tracker-tls
  rules:
  - host: api.carbon-tracker.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: carbon-api-service
            port:
              number: 80
EOF

    # HPA
    cat > k8s/base/hpa.yaml << EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: carbon-api-hpa
  namespace: carbon-tracker
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: carbon-api
  minReplicas: 3
  maxReplicas: 20
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
EOF
}

setup_monitoring() {
    log_info "Setting up monitoring and observability..."
    
    # Prometheus configuration
    cat > config/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "carbon_tracker_rules.yml"

scrape_configs:
  - job_name: 'carbon-api'
    static_configs:
      - targets: ['carbon-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka:9092']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    # Grafana provisioning
    mkdir -p config/grafana/provisioning/{dashboards,datasources}
    
    cat > config/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    log_success "Monitoring configuration created!"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure with Terraform..."
    
    cd terraform
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -var="environment=${ENVIRONMENT}" -var="region=${REGION}"
    
    # Apply (with confirmation)
    read -p "Do you want to apply the Terraform configuration? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        terraform apply -var="environment=${ENVIRONMENT}" -var="region=${REGION}" -auto-approve
        log_success "Infrastructure deployed successfully!"
    else
        log_warning "Infrastructure deployment skipped."
    fi
    
    cd ..
}

start_local_services() {
    log_info "Starting local services with Docker Compose..."
    
    # Pull latest images
    docker-compose pull
    
    # Build custom images
    docker-compose build
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 30
    
    # Check service health
    check_service_health
    
    log_success "All services are running!"
}

check_service_health() {
    local services=("kafka:9092" "influxdb:8086" "redis:6379" "postgres:5432")
    
    for service in "${services[@]}"; do
        local host=$(echo $service | cut -d: -f1)
        local port=$(echo $service | cut -d: -f2)
        
        log_info "Checking $host:$port..."
        
        if timeout 10 bash -c "</dev/tcp/$host/$port"; then
            log_success "$host:$port is ready"
        else
            log_warning "$host:$port is not ready yet"
        fi
    done
}

setup_sample_data() {
    log_info "Setting up sample data..."
    
    # Wait for API to be ready
    sleep 10
    
    # Create sample data script
    cat > scripts/create_sample_data.py << 'EOF'
import requests
import json
import time
from datetime import datetime

API_BASE = "http://localhost:8000"

# Sample emission data
sample_data = [
    {
        "device_id": "smart_meter_001",
        "emission_type": "electricity_grid",
        "consumption_value": 25.5,
        "consumption_unit": "kWh",
        "location": {"lat": 40.7128, "lng": -74.0060},
        "metadata": {"building": "office_main"}
    },
    {
        "device_id": "vehicle_001",
        "emission_type": "gasoline",
        "consumption_value": 8.2,
        "consumption_unit": "liters",
        "location": {"lat": 40.7589, "lng": -73.9851},
        "metadata": {"vehicle_type": "sedan"}
    }
]

for data in sample_data:
    try:
        response = requests.post(f"{API_BASE}/emissions/calculate", json=data)
        if response.status_code == 200:
            print(f"✓ Created sample data for {data['device_id']}")
        else:
            print(f"✗ Failed to create data for {data['device_id']}: {response.text}")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    time.sleep(1)

print("Sample data creation completed!")
EOF

    python scripts/create_sample_data.py
    
    log_success "Sample data created!"
}

show_access_info() {
    log_success "Carbon Footprint Tracker is now running!"
    echo
    echo "🌐 Access URLs:"
    echo "  API Documentation: http://localhost:8000/docs"
    echo "  Grafana Dashboard: http://localhost:3000 (admin/carbontracker123)"
    echo "  Prometheus: http://localhost:9090"
    echo "  InfluxDB: http://localhost:8086"
    echo "  MLflow: http://localhost:5000"
    echo "  Kibana: http://localhost:5601"
    echo
    echo "📊 Sample API Calls:"
    echo "  curl http://localhost:8000/health"
    echo "  curl http://localhost:8000/analytics/dashboard"
    echo
    echo "🔧 Management Commands:"
    echo "  docker-compose logs -f [service]  # View logs"
    echo "  docker-compose stop              # Stop services"
    echo "  docker-compose down              # Stop and remove containers"
    echo
}

cleanup() {
    log_info "Cleaning up..."
    docker-compose down -v
    log_success "Cleanup completed!"
}

# Main execution
main() {
    echo "🌍 Carbon Footprint Tracker Setup Script"
    echo "=========================================="
    
    case "${1:-setup}" in
        "setup")
            check_prerequisites
            setup_local_environment
            setup_docker_environment
            setup_kubernetes_configs
            setup_monitoring
            start_local_services
            setup_sample_data
            show_access_info
            ;;
        "infrastructure")
            check_prerequisites
            deploy_infrastructure
            ;;
        "start")
            start_local_services
            show_access_info
            ;;
        "stop")
            docker-compose stop
            ;;
        "cleanup")
            cleanup
            ;;
        "health")
            check_service_health
            ;;
        *)
            echo "Usage: $0 {setup|infrastructure|start|stop|cleanup|health}"
            echo
            echo "Commands:"
            echo "  setup         - Complete setup (default)"
            echo "  infrastructure - Deploy cloud infrastructure"
            echo "  start         - Start local services"
            echo "  stop          - Stop local services"
            echo "  cleanup       - Stop and remove all containers"
            echo "  health        - Check service health"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"