#!/bin/bash
# =============================================
# 🚀 TFT Data Ingestion - Cloud Deployment Script
# =============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="tft-data-ingestion"
VERSION="v1.0"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-wedhazo}"

echo -e "${BLUE}🚀 Starting deployment of TFT Data Ingestion Microservice${NC}"
echo "=========================================================="

# Function to deploy to Docker Hub
deploy_docker_hub() {
    echo -e "${YELLOW}📦 Building and pushing to Docker Hub...${NC}"
    
    # Build the image
    docker build -t ${DOCKER_REGISTRY}/${SERVICE_NAME}:${VERSION} .
    docker build -t ${DOCKER_REGISTRY}/${SERVICE_NAME}:latest .
    
    # Push to Docker Hub
    docker push ${DOCKER_REGISTRY}/${SERVICE_NAME}:${VERSION}
    docker push ${DOCKER_REGISTRY}/${SERVICE_NAME}:latest
    
    echo -e "${GREEN}✅ Successfully pushed to Docker Hub!${NC}"
}

# Function to deploy to AWS ECS
deploy_aws_ecs() {
    echo -e "${YELLOW}☁️ Deploying to AWS ECS...${NC}"
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        echo -e "${RED}❌ AWS CLI is not installed. Please install it first.${NC}"
        exit 1
    fi
    
    # Build and push to ECR
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    AWS_REGION=${AWS_DEFAULT_REGION:-us-east-1}
    ECR_REPOSITORY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${SERVICE_NAME}"
    
    # Login to ECR
    aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY%/*}
    
    # Create repository if it doesn't exist
    aws ecr create-repository --repository-name ${SERVICE_NAME} --region ${AWS_REGION} 2>/dev/null || true
    
    # Build and push
    docker build -t ${ECR_REPOSITORY}:${VERSION} .
    docker build -t ${ECR_REPOSITORY}:latest .
    docker push ${ECR_REPOSITORY}:${VERSION}
    docker push ${ECR_REPOSITORY}:latest
    
    # Update ECS task definition
    sed "s/ACCOUNT_ID/${AWS_ACCOUNT_ID}/g; s/REGION/${AWS_REGION}/g" aws-ecs-task.json > aws-ecs-task-updated.json
    aws ecs register-task-definition --cli-input-json file://aws-ecs-task-updated.json
    
    echo -e "${GREEN}✅ Successfully deployed to AWS ECS!${NC}"
}

# Function to deploy to Google Cloud Run
deploy_gcloud_run() {
    echo -e "${YELLOW}☁️ Deploying to Google Cloud Run...${NC}"
    
    # Check if gcloud CLI is installed
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ Google Cloud CLI is not installed. Please install it first.${NC}"
        exit 1
    fi
    
    # Get project ID
    PROJECT_ID=$(gcloud config get-value project)
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${RED}❌ No Google Cloud project selected. Run: gcloud config set project PROJECT_ID${NC}"
        exit 1
    fi
    
    # Build and push to Google Container Registry
    GCR_IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${VERSION}"
    
    docker build -t ${GCR_IMAGE} .
    docker push ${GCR_IMAGE}
    
    # Deploy to Cloud Run
    gcloud run deploy ${SERVICE_NAME} \
        --image ${GCR_IMAGE} \
        --platform managed \
        --region us-central1 \
        --port 8001 \
        --memory 2Gi \
        --cpu 1 \
        --min-instances 1 \
        --max-instances 10 \
        --timeout 300 \
        --concurrency 100 \
        --allow-unauthenticated
    
    echo -e "${GREEN}✅ Successfully deployed to Google Cloud Run!${NC}"
}

# Function to deploy to Azure Container Instances
deploy_azure_aci() {
    echo -e "${YELLOW}☁️ Deploying to Azure Container Instances...${NC}"
    
    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        echo -e "${RED}❌ Azure CLI is not installed. Please install it first.${NC}"
        exit 1
    fi
    
    # Configuration
    RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-tft-rg}"
    REGISTRY_NAME="${AZURE_REGISTRY_NAME:-tftregistry}"
    LOCATION="${AZURE_LOCATION:-eastus}"
    
    # Create resource group if it doesn't exist
    az group create --name ${RESOURCE_GROUP} --location ${LOCATION}
    
    # Create container registry if it doesn't exist
    az acr create --resource-group ${RESOURCE_GROUP} --name ${REGISTRY_NAME} --sku Basic --admin-enabled true
    
    # Build and push to ACR
    az acr build --registry ${REGISTRY_NAME} --image ${SERVICE_NAME}:${VERSION} .
    
    # Deploy to Azure Container Instances
    az container create \
        --resource-group ${RESOURCE_GROUP} \
        --name ${SERVICE_NAME} \
        --image ${REGISTRY_NAME}.azurecr.io/${SERVICE_NAME}:${VERSION} \
        --registry-login-server ${REGISTRY_NAME}.azurecr.io \
        --registry-username ${REGISTRY_NAME} \
        --registry-password $(az acr credential show --name ${REGISTRY_NAME} --query "passwords[0].value" --output tsv) \
        --cpu 1 \
        --memory 2 \
        --ports 8001 \
        --environment-variables SERVICE_PORT=8001 LOG_LEVEL=INFO \
        --restart-policy Always
    
    echo -e "${GREEN}✅ Successfully deployed to Azure Container Instances!${NC}"
}

# Main deployment logic
case "${1:-docker}" in
    "docker"|"dockerhub")
        deploy_docker_hub
        ;;
    "aws"|"ecs")
        deploy_aws_ecs
        ;;
    "gcp"|"gcloud"|"cloudrun")
        deploy_gcloud_run
        ;;
    "azure"|"aci")
        deploy_azure_aci
        ;;
    "all")
        deploy_docker_hub
        echo ""
        deploy_aws_ecs
        echo ""
        deploy_gcloud_run
        echo ""
        deploy_azure_aci
        ;;
    *)
        echo "Usage: $0 [docker|aws|gcp|azure|all]"
        echo ""
        echo "Available deployment targets:"
        echo "  docker/dockerhub  - Deploy to Docker Hub"
        echo "  aws/ecs          - Deploy to AWS ECS"
        echo "  gcp/cloudrun     - Deploy to Google Cloud Run"
        echo "  azure/aci        - Deploy to Azure Container Instances"
        echo "  all              - Deploy to all platforms"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo "=========================================================="
echo -e "${BLUE}Service: ${SERVICE_NAME}${NC}"
echo -e "${BLUE}Version: ${VERSION}${NC}"
echo -e "${BLUE}Platform: ${1:-docker}${NC}"
