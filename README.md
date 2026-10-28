# Loan Default Risk Checker

End-to-end machine learning project: a tuned XGBoost credit-risk model, served by a FastAPI backend with a web UI.

## What it does
Takes an applicant and loan details and returns the estimated probability of default, a risk band, and an approve/reject decision.

## Model
- Dataset: public credit risk dataset (about 32k loans, 22% defaults)
- Pipeline: imputation, one-hot encoding, XGBoost with class weighting
- Compared against a logistic regression baseline (cross-validated ROC-AUC 0.87 vs 0.94)
- Probabilities calibrated with sigmoid (Platt) scaling
- Decision threshold chosen from the precision-recall curve to maximise F1
- Explainability with SHAP (see Credit_Risk.ipynb)

## Results (held-out test set, 6,305 loans)
| Metric | XGBoost (calibrated) | Logistic regression baseline |
|---|---|---|
| ROC-AUC | 0.949 | 0.87 (5-fold CV) |
| Accuracy | 93.8% | 81.9% |
| Precision (defaults) | 97.9% | 55.8% |
| Recall (defaults) | 72.9% | 78.7% |
| F1 (defaults) | 83.6% | 65.3% |

The decision threshold (0.725) was tuned on this same test set, so these numbers are slightly optimistic. A stricter setup would tune the threshold on a separate validation split.

## Features
- Probability of default with a visual decision line
- Per-prediction explanation of which inputs pushed the score up or down
- Warnings when an input is outside the range the model was trained on
- Input validation (FastAPI and Pydantic) and interactive API docs at `/docs`

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000 (API docs at /docs).

## API
`POST /predict` with the 10 applicant fields. `loan_percent_income` is computed server-side.

## Deploy (Hugging Face Spaces, Docker)
1. Create a new Space, choose the Docker SDK.
2. Upload every file in this folder.
3. The Space builds from the Dockerfile and serves on port 7860.
