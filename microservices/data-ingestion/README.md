# 🚀 TFT Data Ingestion Microservice - Cloud Deployment

## Overview
The Data Ingestion microservice handles real-time data collection from Polygon.io and Reddit, publishing processed data to Kafka topics for consumption by other microservices.

## Features
- ✅ **Real-time Market Data**: Polygon.io OHLCV, news, and fundamentals
- ✅ **Social Sentiment**: Reddit posts and comments analysis
- ✅ **Kafka Publishing**: Structured data streaming
- ✅ **Redis Caching**: High-performance data caching
- ✅ **Health Monitoring**: Comprehensive health checks
- ✅ **Auto-scaling**: Cloud-native scaling capabilities

## Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Polygon.io    │    │     Reddit      │    │   Data Sources  │
│   Market Data   │    │   Sentiment     │    │   (Future)      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────────┐
                    │   Data Ingestion Service   │
                    │   - Rate Limiting          │
                    │   - Error Handling         │
                    │   - Data Validation        │
                    │   - Caching (Redis)        │
                    └─────────────┬───────────────┘
                                 │
                    ┌─────────────▼───────────────┐
                    │        Kafka Topics        │
                    │   - market_data            │
                    │   - sentiment_data         │
                    │   - news_data              │
                    └─────────────────────────────┘
```

## Quick Start

### 1. **Docker Hub Deployment** (Easiest)
```bash
# Deploy to Docker Hub
./deploy.sh docker

# Or using docker-compose for local testing
docker-compose -f docker-compose.cloud.yml up -d
```

### 2. **AWS ECS Deployment**
```bash
# Prerequisites: AWS CLI configured
aws configure

# Deploy to AWS ECS
./deploy.sh aws
```

### 3. **Google Cloud Run Deployment**
```bash
# Prerequisites: Google Cloud CLI configured
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Deploy to Google Cloud Run
./deploy.sh gcp
```

### 4. **Azure Container Instances**
```bash
# Prerequisites: Azure CLI configured
az login

# Deploy to Azure
./deploy.sh azure
```

## Configuration

### Environment Variables
| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `POLYGON_API_KEY` | Polygon.io API key | ✅ | - |
| `POLYGON_SECRET_KEY` | Polygon.io secret key | ✅ | - |
| `REDDIT_CLIENT_ID` | Reddit API client ID | ✅ | - |
| `REDDIT_CLIENT_SECRET` | Reddit API secret | ✅ | - |
| `REDDIT_USER_AGENT` | Reddit user agent | ✅ | `Kironix_Trading_Sentiment_v1.0` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka cluster endpoints | ✅ | `kafka:9092` |
| `REDIS_HOST` | Redis server host | ✅ | `redis` |
| `REDIS_PORT` | Redis server port | ❌ | `6379` |
| `SERVICE_PORT` | Service listening port | ❌ | `8001` |
| `LOG_LEVEL` | Logging level | ❌ | `INFO` |
| `RATE_LIMIT_SECONDS` | API rate limiting | ❌ | `0.5` |
| `MAX_RETRIES` | Maximum retry attempts | ❌ | `3` |

### Secrets Management

#### AWS Secrets Manager
```bash
# Create secrets in AWS
aws secretsmanager create-secret --name "tft/polygon-api-key" --secret-string "your-api-key"
aws secretsmanager create-secret --name "tft/reddit-client-id" --secret-string "your-client-id"
```

#### Google Cloud Secret Manager
```bash
# Create secrets in Google Cloud
echo -n "your-api-key" | gcloud secrets create polygon-api-key --data-file=-
echo -n "your-client-id" | gcloud secrets create reddit-client-id --data-file=-
```

#### Azure Key Vault
```bash
# Create secrets in Azure Key Vault
az keyvault secret set --vault-name tft-keyvault --name polygon-api-key --value "your-api-key"
az keyvault secret set --vault-name tft-keyvault --name reddit-client-id --value "your-client-id"
```

## API Endpoints

### Health Check
```bash
GET /health
```
**Response:**
```json
{
  "status": "healthy",
  "service": "data-ingestion",
  "version": "v1.0",
  "timestamp": "2025-08-07T20:00:00Z",
  "dependencies": {
    "polygon": "connected",
    "reddit": "connected",
    "kafka": "connected",
    "redis": "connected"
  }
}
```

### Data Collection Status
```bash
GET /status
```
**Response:**
```json
{
  "active_symbols": ["AAPL", "GOOGL", "MSFT"],
  "kafka_topics": ["tft_market_data", "tft_sentiment_data"],
  "rate_limits": {
    "polygon": "4/5 requests used",
    "reddit": "100/100 requests remaining"
  },
  "cache_stats": {
    "redis_connections": 5,
    "cache_hits": 1247,
    "cache_misses": 23
  }
}
```

### Manual Data Collection
```bash
POST /collect/symbols
Content-Type: application/json

{
  "symbols": ["AAPL", "MSFT", "GOOGL"],
  "data_types": ["ohlcv", "news", "sentiment"]
}
```

## Kafka Topics Produced

### 1. `tft_market_data`
Real-time market data from Polygon.io
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-08-07T20:00:00Z",
  "data": {
    "open": 185.50,
    "high": 187.25,
    "low": 184.75,
    "close": 186.80,
    "volume": 45000000,
    "vwap": 186.12
  },
  "source": "polygon"
}
```

### 2. `tft_sentiment_data`
Social media sentiment analysis
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-08-07T20:00:00Z",
  "data": {
    "sentiment_score": 0.65,
    "confidence": 0.89,
    "mention_count": 147,
    "source_breakdown": {
      "reddit": 89,
      "twitter": 58
    }
  },
  "source": "reddit"
}
```

### 3. `tft_news_data`
Financial news and events
```json
{
  "symbol": "AAPL",
  "timestamp": "2025-08-07T20:00:00Z",
  "data": {
    "headline": "Apple Reports Strong Q3 Earnings",
    "summary": "...",
    "sentiment": 0.72,
    "impact_score": 0.85,
    "publisher": "Reuters"
  },
  "source": "polygon"
}
```

## Monitoring & Observability

### Metrics Exposed
- HTTP request latency and throughput
- Kafka producer metrics
- Redis connection pool stats
- API rate limit utilization
- Error rates by source

### Logging
Structured JSON logs with correlation IDs:
```json
{
  "timestamp": "2025-08-07T20:00:00Z",
  "level": "INFO",
  "service": "data-ingestion",
  "trace_id": "abc123",
  "message": "Successfully collected data for AAPL",
  "metadata": {
    "symbol": "AAPL",
    "data_type": "ohlcv",
    "latency_ms": 125
  }
}
```

## Scaling & Performance

### Auto-scaling Configuration
- **CPU Threshold**: 70%
- **Memory Threshold**: 80%
- **Min Replicas**: 1
- **Max Replicas**: 10
- **Target Requests/Second**: 100

### Performance Benchmarks
- **Throughput**: 500 symbols/minute
- **Latency**: <200ms average
- **Memory Usage**: ~1GB per instance
- **CPU Usage**: ~0.5 vCPU per instance

## Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export POLYGON_API_KEY="your-key"
export REDDIT_CLIENT_ID="your-id"

# Run locally
python main.py
```

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Load testing
./load_test.sh
```

## Deployment Automation

### CI/CD Pipeline
The service includes GitHub Actions workflows for automated deployment:

1. **Build**: Dockerfile validation and image build
2. **Test**: Unit and integration tests
3. **Security**: Container vulnerability scanning
4. **Deploy**: Multi-cloud deployment with rollback capability

### Rollback Procedure
```bash
# AWS ECS rollback
aws ecs update-service --cluster tft-cluster --service data-ingestion --task-definition tft-data-ingestion:PREVIOUS_REVISION

# Google Cloud Run rollback
gcloud run services update-traffic tft-data-ingestion --to-revisions=PREVIOUS_REVISION=100

# Docker Compose rollback
docker-compose -f docker-compose.cloud.yml down
docker-compose -f docker-compose.cloud.yml up -d
```

## Security

### Network Security
- All external communication over HTTPS/TLS
- Internal service mesh with mTLS
- API key rotation supported
- VPC isolation in cloud deployments

### Data Security
- API keys stored in secure secret managers
- Data encryption in transit and at rest
- PII scrubbing for social media data
- Audit logging for all data access

## Troubleshooting

### Common Issues

#### 1. Polygon.io Rate Limiting
**Symptom**: HTTP 429 errors
**Solution**: 
```bash
# Check rate limit configuration
kubectl logs deployment/tft-data-ingestion | grep "rate_limit"

# Adjust rate limiting
kubectl set env deployment/tft-data-ingestion RATE_LIMIT_SECONDS=1.0
```

#### 2. Kafka Connection Issues
**Symptom**: Unable to publish messages
**Solution**:
```bash
# Verify Kafka connectivity
kubectl exec -it deployment/tft-data-ingestion -- curl kafka:9092

# Check Kafka topic creation
kubectl exec -it kafka-0 -- kafka-topics.sh --list --bootstrap-server localhost:9092
```

#### 3. Redis Connection Pool Exhaustion
**Symptom**: Redis connection timeouts
**Solution**:
```bash
# Monitor Redis connections
kubectl exec -it deployment/tft-data-ingestion -- redis-cli info clients

# Scale Redis or adjust pool size
kubectl set env deployment/tft-data-ingestion REDIS_POOL_SIZE=20
```

## Support

For technical support and questions:
- 📧 **Email**: support@tft-trading.com
- 💬 **Discord**: [TFT Trading Community](https://discord.gg/tft-trading)
- 🐛 **Issues**: [GitHub Issues](https://github.com/wedhazo/TFT/issues)

---

## Next Steps

✅ **Data Ingestion** - **READY FOR CLOUD DEPLOYMENT**  
⏳ **Sentiment Engine** - Next microservice  
⏳ **TFT Predictor** - Coming next  
⏳ **Trading Engine** - Coming next  
⏳ **Orchestrator** - Final integration
