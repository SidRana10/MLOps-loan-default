# MLOps Loan Prediction Project

An end-to-end MLOps pipeline for a loan approval prediction model. The project covers data versioning, hyperparameter-tuned model training with experiment tracking, a FastAPI inference service, containerized deployment to Kubernetes (AWS EKS), CI/CD via GitHub Actions, and a Streamlit-based data drift/quality monitoring dashboard.

## Features

- **XGBoost classification pipeline** with scikit-learn `Pipeline` for preprocessing and modeling
- **Hyperparameter optimization** using `hyperopt` (TPE algorithm) across multiple XGBoost parameters
- **Experiment tracking & model registry** with MLflow, selecting the best run by F1 score
- **Custom preprocessing transformers**: domain feature engineering, mean/mode imputation, label encoding, log transforms, and feature dropping
- **FastAPI inference service** with three endpoints: single prediction (JSON), single prediction (query params/UI), and batch prediction (CSV upload)
- **Batch predictions exported to S3** for downstream monitoring, organized by date
- **Prometheus instrumentation** for API metrics via `prometheus-fastapi-instrumentator`
- **Data drift & data quality monitoring** dashboard built with Streamlit and Evidently AI, comparing baseline vs. recent production data from S3
- **Data & model versioning** with DVC, backed by an S3 remote
- **Automated tests** validating prediction output type and correctness
- **CI/CD pipeline** that builds a Docker image, pushes it to Amazon ECR, and deploys to an AWS EKS cluster on every push to `main`

## Architecture

```
                 ┌──────────────────┐
                 │   DVC (S3 remote) │  <- datasets & trained models
                 └────────┬─────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  training_pipeline.py    │
              │  (Hyperopt + XGBoost)    │
              │  Tracks runs in MLflow   │
              └────────┬────────────────┘
                          │ best model (by F1)
                          ▼
              ┌────────────────────────┐
              │     FastAPI (main.py)    │
              │  /prediction_api          │
              │  /prediction_ui           │
              │  /batch_prediction        │
              │  Prometheus metrics       │
              └────────┬────────────────┘
                          │ batch results
                          ▼
              ┌────────────────────────┐
              │   S3 (datadrift/...)     │
              └────────┬────────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │ drift_monitoring/app_v1  │
              │ (Streamlit + Evidently)  │
              │ Data Drift & Quality     │
              └────────────────────────┘
```

The service is containerized with Docker and deployed to an AWS EKS cluster via `deployment.yml` and `service.yml`, with the CI/CD workflow in `.github/workflows/main.yml` building/pushing the image to ECR and applying the manifests.

## Project Structure

```
.
├── main.py                          # FastAPI app: prediction & batch prediction endpoints
├── prediction_model/
│   ├── config/
│   │   └── config.py                 # Paths, feature lists, MLflow/S3 settings
│   ├── processing/
│   │   ├── data_handling.py          # Dataset loading utilities
│   │   └── preprocessing.py          # Custom sklearn transformers
│   ├── pipeline.py                   # Preprocessing pipeline definition
│   ├── training_pipeline.py          # Hyperopt-tuned XGBoost training + MLflow logging
│   ├── predict.py                    # Loads best MLflow model & generates predictions
│   ├── datasets/                      # Train/test data (DVC-tracked)
│   ├── trained_models/                # Model artifacts (DVC-tracked)
│   └── VERSION
├── drift_monitoring/
│   └── app_v1.py                     # Streamlit dashboard for data drift & data quality
├── tests/
│   └── test_prediction.py            # Prediction output tests
├── .github/workflows/main.yml        # CI/CD: build, push to ECR, deploy to EKS
├── deployment.yml                    # Kubernetes Deployment manifest
├── service.yml                       # Kubernetes Service (LoadBalancer) manifest
├── Dockerfile
├── requirements.txt
└── .dvc/                              # DVC configuration (S3 remote)
```

## Prerequisites

- Python 3.10+
- AWS account with access to S3 (for DVC remote storage and batch prediction uploads)
- An MLflow tracking server (the project points to a remote tracking URI by default)
- Docker (for containerized builds/deployment)
- `kubectl` and access to an AWS EKS cluster (for production deployment)

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd MLOps_Project
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure AWS credentials

The project uses `boto3` for S3 access (batch prediction uploads and DVC remote). Configure credentials via environment variables or the AWS CLI:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

### 4. Pull data and model artifacts with DVC

```bash
pip install dvc[s3]
dvc pull
```

### 5. Configure MLflow tracking

By default, `prediction_model/config/config.py` points to a remote MLflow tracking server (`TRACKING_URI`). Update this to your own MLflow server URI (e.g. `http://localhost:5000` for a local instance) if needed.

## Usage

### Train the model

Runs hyperparameter search with Hyperopt over an XGBoost classifier, logging each run (metrics, parameters, and model artifact) to MLflow:

```bash
python prediction_model/training_pipeline.py
```

The best model (by F1 score) is automatically selected at prediction time via MLflow's run search.

### Run the API server

```bash
python main.py
```

or with Uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8005
```

### Endpoints

| Method | Endpoint              | Description                                              |
|--------|-----------------------|-----------------------------------------------------------|
| GET    | `/`                    | Welcome / health check message                            |
| POST   | `/prediction_api`      | Single prediction from a JSON payload                     |
| POST   | `/prediction_ui`       | Single prediction via query/form parameters (UI-friendly) |
| POST   | `/batch_prediction`    | Upload a CSV, get predictions appended as a downloadable CSV (also uploaded to S3) |
| GET    | `/metrics`             | Prometheus metrics (via `prometheus-fastapi-instrumentator`) |

**Example: single prediction**

```bash
curl -X POST http://localhost:8005/prediction_api \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Married": "Yes",
    "Dependents": "0",
    "Education": "Graduate",
    "Self_Employed": "No",
    "ApplicantIncome": 5000,
    "CoapplicantIncome": 0,
    "LoanAmount": 150,
    "Loan_Amount_Term": 360,
    "Credit_History": 1,
    "Property_Area": "Urban"
  }'
```

**Response:**

```json
{"status": "Approved"}
```

**Example: batch prediction**

```bash
curl -X POST http://localhost:8005/batch_prediction \
  -F "file=@loan_applications.csv" \
  --output predictions.csv
```

The uploaded CSV must contain all columns listed in `FEATURES` in `prediction_model/config/config.py`. Results are returned as a CSV with an added `Prediction` column and also stored in S3 under `datadrift/<date>/`.

### Data drift & quality monitoring

Launch the Streamlit dashboard to compare a baseline dataset against the most recent batch-prediction data uploaded to S3:

```bash
streamlit run drift_monitoring/app_v1.py
```

The dashboard provides two views, Data Drift and Data Quality, powered by Evidently AI report presets.

## Testing

```bash
pytest -v tests/test_prediction.py
```

Tests validate that predictions are non-null, returned as strings, and correct for a known sample.

## Docker

Build and run the container locally:

```bash
docker build \
  --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  -t mlops-loan-prediction .

docker run -p 8005:8005 mlops-loan-prediction
```

The Dockerfile pulls DVC-tracked data/models, runs the training pipeline, and executes the test suite as part of the image build.

## CI/CD & Deployment

On every push or pull request to `main`, the GitHub Actions workflow (`.github/workflows/main.yml`):

1. Checks out the code and configures AWS credentials
2. Logs in to Amazon ECR
3. Builds the Docker image (running training and tests as part of the build) and pushes it to ECR
4. Updates the kubeconfig for the `mlops` EKS cluster
5. Applies `deployment.yml` and `service.yml` to deploy/update the service on Kubernetes

The Kubernetes Service exposes the app via a `LoadBalancer` on port 80, forwarding to container port 8005.

## Tech Stack

- **Modeling**: scikit-learn, XGBoost, Hyperopt
- **Experiment Tracking**: MLflow
- **API**: FastAPI, Uvicorn, Gunicorn
- **Monitoring**: Prometheus, Streamlit, Evidently AI
- **Data/Model Versioning**: DVC (S3 remote)
- **Cloud & Deployment**: AWS (S3, ECR, EKS), Docker, Kubernetes
- **CI/CD**: GitHub Actions
- **Testing**: pytest
