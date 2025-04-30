# Predictive Maintenance API

A FastAPI-based service for predicting machine failures based on sensor data.

## Features

- Machine failure prediction using Random Forest Classifier
- RESTful API endpoints
- Prometheus metrics for monitoring
- Structured logging
- Docker support
- Health checks
- CORS support

## API Endpoints

- `GET /`: Welcome message
- `POST /predict`: Make a prediction
- `GET /health`: Health check endpoint
- `GET /metrics`: Prometheus metrics
- `GET /docs`: Swagger documentation
- `GET /redoc`: ReDoc documentation

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- PostgreSQL (for production)

## Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn app:app --reload
```

## Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up -d
```

2. Check the logs:
```bash
docker-compose logs -f
```

## Environment Variables

Create a `.env` file with the following variables:

```env
PORT=8000
HOST=0.0.0.0
DEBUG=False
MODEL_PATH=model.joblib
SCALER_PATH=scaler.joblib
DATA_PATH=predictive_maintenance.csv
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOG_RETENTION_DAYS=10
LOG_ROTATION_SIZE=500MB
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

## Monitoring

The application exposes Prometheus metrics at `/metrics`. You can use these metrics with tools like Grafana for monitoring.

## Logging

Logs are stored in the `logs` directory with rotation and retention policies.

## Health Checks

The `/health` endpoint can be used for health checks in container orchestration systems.

## License

MIT 