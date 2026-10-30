# CredShap: Loan Default Risk Assessment

**Live demo: https://credshap-2.onrender.com**

The demo runs on a free hosting plan, so the first visit after a quiet period can take about a minute to wake up.

An end-to-end machine learning project. A tuned, probability-calibrated XGBoost model estimates the chance that a loan defaults, and a FastAPI backend serves it through a web interface that also explains each score.

## What it does
Enter an applicant and loan details. The app returns:
- the estimated probability of default
- a risk band (low, medium, high) and an approve or reject decision
- the five factors that pushed the score up or down the most (SHAP values)
- a warning when an input lies outside the range the model was trained on

A second tab, **Model details**, shows the test results, the confusion matrix, how the decision threshold trades one kind of error for another, and the model comparison.

## Model
- Dataset: public credit risk dataset (about 32,000 loans after cleaning, 22% defaults)
- Preprocessing and model live in one scikit-learn pipeline, so the same steps run at training and prediction time
- Compared with a logistic regression baseline using 5-fold cross-validation (ROC-AUC 0.87 vs 0.94)
- XGBoost tuned with 150 random search trials, scored on average precision
- Probabilities calibrated with sigmoid (Platt) scaling
- Decision threshold (0.725) chosen from the precision-recall curve to maximise F1
- Explanations: XGBoost's built-in SHAP values, averaged across the 5 calibration folds, with one-hot columns grouped back to the original field

## Results (held-out test set, 6,305 loans)
| Metric | XGBoost (tuned, calibrated) | Logistic regression baseline |
|---|---|---|
| ROC-AUC | 0.949 | 0.87 (5-fold CV) |
| Accuracy | 93.8% | 81.9% |
| Precision (defaults) | 97.9% | 55.8% |
| Recall (defaults) | 72.9% | 78.7% |
| F1 (defaults) | 83.6% | 65.3% |

On the test set the model wrongly rejects 21 good loans and misses 369 defaults. A lower threshold catches more defaults but rejects more good borrowers (see the Model details tab).

## Limitations
- The threshold was tuned on the same test set used for the results above, so real performance is likely slightly lower. A stricter setup would use a separate validation split.
- The data is a public dataset, not a lender's records, and was not checked for fairness across groups.
- This is a portfolio project, not a lending tool.

## Tech stack
Python, scikit-learn, XGBoost, FastAPI, Pydantic, Docker. The interface is plain HTML, CSS and JavaScript. Deployed on Render.

## Run locally
Use Python 3.12 (the pinned library versions do not install on Python 3.14).
```bash
python -m venv .venv
.venv\Scripts\activate        # on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
Open http://127.0.0.1:8000. Interactive API docs are at `/docs`.

## API
`POST /predict` with the 10 applicant fields. `loan_percent_income` is computed on the server. `GET /api/model-info` returns the metrics shown on the Model details tab.

## Deploy
The included `Dockerfile` works on Render (choose the Docker runtime) and on any other host that builds from a Dockerfile.

## Project files
- `main.py`: FastAPI app, prediction and explanation logic
- `static/index.html`: the web interface
- `credit_risk_model.pkl`, `best_threshold.pkl`: the trained model and its decision threshold
- `Credit_Risk.ipynb`: data cleaning, model comparison, tuning and evaluation
- `metrics.json`: test results shown in the app
