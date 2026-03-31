# 🛡️ Insurance Premium Predictor

A full-stack machine learning application that predicts insurance premium categories — **Low**, **Medium**, or **High** — based on a user's personal, financial, and lifestyle profile. Built with a **FastAPI** backend, a **scikit-learn** classification model, and a dark-themed interactive frontend.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [ML Model](#ml-model)
- [Dataset & Features](#dataset--features)
- [API Endpoints](#api-endpoints)
- [Input Schema](#input-schema)
- [Output Schema](#output-schema)
- [Frontend](#frontend)
- [Getting Started](#getting-started)
- [Running with Docker](#running-with-docker)
- [Tech Stack](#tech-stack)

---

## Overview

This project exposes a REST API that accepts user profile data and returns a predicted insurance premium tier along with class-level probability scores. The prediction is powered by a pre-trained scikit-learn model (`model.pkl`) that was trained on a synthetic Indian insurance dataset.

The application serves a polished, self-contained HTML frontend directly from the FastAPI root endpoint, making it usable out of the box without any separate frontend deployment.

---

## Project Structure

```
insurance-premium-prediction-fastapi/
│
├── app.py                        # FastAPI application & route definitions
├── frontend.html                 # Interactive single-page UI
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition
│
├── model/
│   ├── model.pkl                 # Serialised scikit-learn classifier
│   └── predict.py                # Model loading & inference logic
│
├── schema/
│   ├── user_input.py             # Pydantic input model with computed fields
│   └── prediction_response.py    # Pydantic response model
│
└── config/
    └── city_tier.py              # City-to-tier mapping (Tier 1 / 2 / 3)
```

---

## ML Model

| Property | Detail |
|---|---|
| **Version** | `1.0.0` |
| **File** | `model/model.pkl` |
| **Framework** | scikit-learn 1.6.1 |
| **Type** | Multi-class probabilistic classifier (supports `predict_proba`) — e.g., Random Forest / Gradient Boosting |
| **Output classes** | `Low`, `Medium`, `High` |
| **Serialisation** | Python `pickle` |

The model is loaded once at startup inside `model/predict.py` and reused across all requests. Inference is performed on a single-row `pandas` DataFrame constructed from the six engineered features.

### Inference Pipeline

```
Raw user input (7 fields)
        │
        ▼
  Pydantic UserInput model
  (validation + computed fields)
        │
        ├─► bmi              = weight / height²
        ├─► age_group        = young / adult / middle_aged / senior
        ├─► lifestyle_risk   = low / medium / high
        └─► city_tier        = 1 / 2 / 3
        │
        ▼
  6-feature DataFrame → model.predict() + model.predict_proba()
        │
        ▼
  { predicted_category, confidence, class_probabilities }
```

---

## Dataset & Features

The model was trained on a synthetic dataset representing Indian insurance applicants. The dataset is designed to reflect real-world premium-pricing factors used by Indian insurers.

### Raw Input Fields

| Field | Type | Description | Constraints |
|---|---|---|---|
| `age` | `int` | Age of the applicant | 1 – 119 |
| `weight` | `float` | Weight in kilograms | > 0 |
| `height` | `float` | Height in metres | 0.5 – 2.49 |
| `income_lpa` | `float` | Annual income in Lakhs Per Annum | > 0 |
| `smoker` | `bool` | Whether the applicant smokes | `true` / `false` |
| `city` | `str` | City of residence | Any Indian city |
| `occupation` | `str` (enum) | Applicant's occupation | See values below |

**Valid occupation values:** `retired`, `freelancer`, `student`, `government_job`, `business_owner`, `unemployed`, `private_job`

### Engineered / Computed Features

These four features are **automatically derived** by the Pydantic model before inference — they are never sent by the client directly.

| Feature | Derivation Logic |
|---|---|
| `bmi` | `weight / height²` |
| `age_group` | `young` (<25) · `adult` (<45) · `middle_aged` (<60) · `senior` (≥60) |
| `lifestyle_risk` | `high` if smoker & BMI > 30 · `medium` if smoker OR BMI > 27 · `low` otherwise |
| `city_tier` | `1` = Metro (Mumbai, Delhi, Bangalore…) · `2` = Tier-2 cities · `3` = All others |

### Target Classes

| Class | Meaning |
|---|---|
| `Low` | Low premium bracket — lower risk profile |
| `Medium` | Mid-range premium — moderate risk indicators |
| `High` | High premium bracket — elevated risk factors |

### City Tier Reference

**Tier 1 (7 cities):** Mumbai, Delhi, Bangalore, Chennai, Kolkata, Hyderabad, Pune

**Tier 2 (48 cities):** Jaipur, Chandigarh, Indore, Lucknow, Surat, Nagpur, Vadodara, Coimbatore, Bhopal, Visakhapatnam, and others across India.

**Tier 3:** All remaining cities.

---

## API Endpoints

### `GET /`
Serves the `frontend.html` interactive UI directly in the browser.

### `GET /health`
Returns backend status and model version.

```json
{
  "status": "OK",
  "version": "1.0.0",
  "model_loaded": true
}
```

### `POST /predict`
Accepts user profile JSON and returns the predicted premium category.

---

## Input Schema

**`POST /predict`** — Request Body

```json
{
  "age": 34,
  "weight": 78.5,
  "height": 1.72,
  "income_lpa": 12.5,
  "smoker": false,
  "city": "Pune",
  "occupation": "private_job"
}
```

> **Note:** `bmi`, `age_group`, `lifestyle_risk`, and `city_tier` are computed server-side from the above fields. Do not include them in the request.

---

## Output Schema

**`POST /predict`** — Response Body (wrapped under a `response` key)

```json
{
  "response": {
    "predicted_category": "Medium",
    "confidence": 0.7812,
    "class_probabilities": {
      "Low": 0.0634,
      "Medium": 0.7812,
      "High": 0.1554
    }
  }
}
```

| Field | Type | Description |
|---|---|---|
| `predicted_category` | `string` | The winning class: `Low`, `Medium`, or `High` |
| `confidence` | `float` | Probability of the predicted class (0–1) |
| `class_probabilities` | `dict` | Probability distribution across all three classes |

---

## Frontend

The `frontend.html` is a fully self-contained dark-themed single-page application served directly by FastAPI at `/`.

### Features

- **Live derived-field preview** — BMI, Age Group, Lifestyle Risk, and City Tier update in real time as you type, mirroring the server-side computation.
- **Backend health indicator** — A status pill in the header polls `/health` every 20 seconds and shows whether the API is online, along with the model version.
- **Field-level validation** — Required fields are highlighted in red if empty on submit, with an error toast at the bottom.
- **Animated results panel** — On a successful prediction, probability bars animate in with the predicted category, confidence score, and a colour-coded badge (green = Low, yellow = Medium, red = High).
- **Responsive layout** — Two-column desktop layout collapses to single-column on mobile.

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/insurance-premium-prediction-fastapi.git
cd insurance-premium-prediction-fastapi

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Then open your browser at:

```
http://127.0.0.1:8000
```

The interactive frontend will load automatically. The FastAPI auto-generated docs are also available at:

```
http://127.0.0.1:8000/docs      # Swagger UI
http://127.0.0.1:8000/redoc     # ReDoc
```

---

## Running with Docker

```bash
# Build the image
docker build -t insurance-premium-predictor .

# Run the container
docker run -p 8000:8000 insurance-premium-predictor
```

Visit `http://localhost:8000` in your browser.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Backend framework** | FastAPI | 0.115.12 |
| **ASGI server** | Uvicorn | 0.34.2 |
| **Data validation** | Pydantic v2 | 2.11.4 |
| **ML framework** | scikit-learn | 1.6.1 |
| **Data processing** | pandas | 2.2.3 |
| **Numerical computing** | NumPy | 2.2.6 |
| **Containerisation** | Docker | — |
| **Frontend** | Vanilla HTML/CSS/JS | — |
| **Python version** | Python | 3.11 |

---

## Example cURL Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "age": 45,
           "weight": 90,
           "height": 1.68,
           "income_lpa": 8.0,
           "smoker": true,
           "city": "Mumbai",
           "occupation": "business_owner"
         }'
```
