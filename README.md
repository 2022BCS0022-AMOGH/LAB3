# Lab 4: Automated Model Deployment Using GitHub Actions (CI/CD for Inference)

## Objective

To build a **metric-gated CI/CD pipeline** for machine learning inference that:

1. Trains a regression model automatically on code push
2. Evaluates the model using **R² (maximize)** and **MSE (minimize)**
3. Deploys the model **only if metrics improve**
4. Builds and pushes a Dockerized FastAPI inference service
5. Blocks deployment for worse models (fail-closed behavior)

This lab demonstrates **production-grade MLOps deployment control**.

---

## Repository Structure

```
LAB3/
├── .github/workflows/
│   └── train_and_deploy.yml
├── dataset/
│   └── winequality-red.csv
├── scripts/
│   └── train.py
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
```

---

## Prerequisites

* GitHub account
* Docker Hub account
* Docker installed locally
* Python 3.11
* GitHub Personal Access Token
* Docker Hub Access Token

---

## Step 1: Configure GitHub Secrets and Variables

### 1.1 Repository Secrets

Go to
**Repository → Settings → Secrets and variables → Actions → Secrets**

Add:

| Name              | Description                               |
| ----------------- | ----------------------------------------- |
| `DOCKER_USERNAME` | Docker Hub username                       |
| `DOCKER_TOKEN`    | Docker Hub access token (Read & Write)    |
| `GH_TOKEN`        | GitHub Personal Access Token (repo scope) |

---

### 1.2 Repository Variables

Go to
**Repository → Settings → Secrets and variables → Actions → Variables**

Add:

| Name       | Initial Value | Purpose      |
| ---------- | ------------- | ------------ |
| `BEST_R2`  | `-9999`       | Baseline R²  |
| `BEST_MSE` | `9999`        | Baseline MSE |

These act as **persistent deployment baselines** across CI runs.

---

## Step 2: Training Script (`scripts/train.py`)

### Supported Models

* Dummy / Zero Regressor (intentional failure)
* Linear Regression
* XGBoost Regressor (final deployed model)

### Contract enforced by CI

The training script **must always produce**:

* `model.pkl`
* `metrics.json` with keys:

```json
{
  "r2": <float>,
  "mse": <float>
}
```

---

## Step 3: CI/CD Workflow (`train_and_deploy.yml`)

### Workflow Trigger

```yaml
on:
  push:
    branches:
      - main
```

Every push to `main` is a deployment candidate.

---

### Job 1: `train` (Continuous Integration)

Steps:

1. Checkout repository
2. Set up Python 3.11
3. Install dependencies
4. Run `scripts/train.py`
5. Upload:

   * `model.pkl`
   * `metrics.json`

Artifacts are passed to the next job.

---

### Job 2: `deploy` (Continuous Deployment)

This job runs **only if training succeeds**.

#### Metric Gate (Fail-Closed)

* Reads current metrics from `metrics.json`
* Reads baseline metrics from GitHub Variables
* Deployment allowed **only if**:

```
current_R2 > BEST_R2  AND  current_MSE < BEST_MSE
```

If the condition fails:

* Job **hard-fails**
* Docker steps are skipped
* Baseline metrics remain unchanged

This guarantees **no accidental deployment**.

---

### Deployment Steps (only after gate passes)

1. Docker login using secrets
2. Build Docker image
3. Push image to Docker Hub
4. Update:

   * `BEST_R2`
   * `BEST_MSE`

---

## Step 4: Inference Service (`app.py`)

* FastAPI-based service
* Loads `model.pkl` at startup
* Exposes `/predict` endpoint
* Returns response in required format:

```json
{
  "name": "Amogh",
  "roll_no": "2022BCS0022",
  "wine_quality": 5
}
```

---

## Step 5: Dockerization

### Dockerfile responsibilities

* Install dependencies
* Copy model and app
* Expose port `8000`
* Run FastAPI using Uvicorn

The image pushed to Docker Hub is **fully deployable**.

---

## Step 6: Post-Deployment Validation (Task 3)

### Pull Image

```bash
docker pull <dockerhub-username>/wine-inference:latest
```

### Run Container

```bash
docker run -p 8000:8000 <dockerhub-username>/wine-inference:latest
```

### Test API

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{ ...wine features... }'
```

---

## Step 7: Validating Metric-Based Deployment

### Case 1: Bad Model (Dummy / Zero Regressor)

* Train job runs
* Deploy job **fails at metric gate**
* Docker image **not updated**
* BEST metrics unchanged

### Case 2: Better Model (XGBoost)

* Train job runs
* Deploy job passes gate
* Docker image built and pushed
* BEST metrics updated

Screenshots of **both cases** are captured for submission.

---

## Key MLOps Concepts Demonstrated

* CI artifacts for model handoff
* Persistent metric baselines
* Fail-closed deployment gates
* Automated Docker publishing
* Reproducible inference deployment

---

## Final Notes

This pipeline ensures:

* Models deploy **only when objectively better**
* No human judgment in deployment decisions
* Full traceability via CI runs and Docker tags

This is **production-grade CI/CD for ML inference**, not a demo script.
