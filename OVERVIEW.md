# Student Placement Prediction: Overview

Trains a RandomForest classifier to predict `Placement` from `CGPA`, `IQ` and `Profile Score`, tracks experiments in MLflow and serves the best model over a FastAPI REST API.

```
data/raw/*.csv --> train_model.py --> MLflow (mlflow.db + mlruns/)
                        |
                        v  best run only
                 models/model.pkl --> predict_model.py --> api.py --> POST /predict
```

## Setup

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training: `src/models/train_model.py`

`train(n_estimators, random_state)` does one MLflow run:

1. Loads `data/raw/student_placement_data.csv`. `Placement` is the target and every other column is a feature.
2. Splits 80/20 with `train_test_split(random_state=42)`, so every run is evaluated on the same test set.
3. Points MLflow at `sqlite:///<root>/mlflow.db` and uses the experiment `Student Placement Prediction`.
4. Fits a `RandomForestClassifier`, then logs the params (`n_estimators`, `random_state`), the `accuracy` metric and the model artifact (under `mlruns/`).

The `__main__` block does the model selection:

1. Runs `train()` three times with different hyperparameters.
2. Queries the experiment with `MlflowClient.search_runs(order_by=["metrics.accuracy DESC"], max_results=1)[0]` to get the best run.
3. Loads that run's model from `runs:/<run_id>/model` and writes it to `models/model.pkl` with joblib.

Only the winning model reaches `models/model.pkl`, and that file is the single hand-off between training and serving.

```bash
python src/models/train_model.py
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

## What MLflow stores: `mlflow.db` + `mlruns/`

The two work as a **pair**:

| | Holds | Used for |
|---|---|---|
| `mlflow.db` (SQLite) | Metadata only: experiments, runs (id, name, status, times, `artifact_uri`), params, metrics, tags, logged models | Listing, searching and comparing runs. This is what `search_runs` and the UI query. |
| `mlruns/<exp_id>/<run_id>/artifacts` | The files: the serialized model | Loading a model, e.g. `runs:/<run_id>/model` |

- Each run row in the DB stores an **absolute path** to its artifacts folder, and the UI and `load_model` follow that path.
- Delete one without the other and you get runs with missing models, or orphaned model files. Moving the project directory also breaks the stored paths.
- Training data is not stored in either place.
- `models/model.pkl` is a separate copy of the best model, made for serving, and the API never touches MLflow.
- `mlflow.db` has about 70 tables, but only `experiments`, `runs`, `params`, `metrics`, `latest_metrics`, `tags` and `logged_models` are used here. The rest back the registry, tracing and evaluation features.
- A `FAILED` run in the DB is a run that crashed (for example, the earlier `skops` error). Its params stay in the DB, so it can still show up in `search_runs`.

```bash
sqlite3 -header mlflow.db "select r.name, r.status, m.value as accuracy from runs r left join metrics m on m.run_uuid = r.run_uuid;"
```

## What's inside `models/model.pkl`

A pickled, fitted `RandomForestClassifier` (saved with `joblib`):

- **Contains:** the learned trees (node splits and leaf values), hyperparameters, `classes_` and `feature_names_in_` (`CGPA`, `IQ`, `Profile Score`).
- **Doesn't contain:** training data, accuracy or any preprocessing code.
- Loading needs a compatible scikit-learn version, so pin it for serving.
- Input columns must match `feature_names_in_` exactly, or `predict` raises an error.
- Pickle can execute code on load, so only load files you trained yourself.

## Serving: `src/models/predict_model.py` and `src/api.py`

**`predict_model.py`** is the inference layer and knows nothing about HTTP:

- `load_model()` loads `models/model.pkl` once and caches it with `lru_cache`. It raises `FileNotFoundError` if the model hasn't been trained yet.
- `predict(features_dict)` raises `ValueError` on empty input. Otherwise it builds a one-row DataFrame, calls `model.predict` and returns a plain Python value.

**`api.py`** is the HTTP layer:

- `POST /predict` takes `{"data": {...features...}}` and calls `predict(input.data)`.
- It maps exceptions to status codes: `FileNotFoundError` becomes **503** (model not trained) and `ValueError` becomes **400** (bad input).
- It appends every request and prediction to `logs/predictions.log`.
- `GET /` is a health check.

The feature keys must match the training columns exactly, including the space in `Profile Score`.

```bash
uvicorn src.api:app --reload   # run from the project root
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": {"CGPA": 9.1, "IQ": 134, "Profile Score": 45}}'
# {"prediction": "Yes"}
```

Interactive docs are at `/docs`.

## Notes

- `load_model()` is cached, so **restart the server after retraining** to pick up a new `model.pkl`.
- The `search_runs` query covers every run ever logged in `mlflow.db`, so the "best" model may come from an older session, and tied accuracies are broken arbitrarily.
- Don't commit `.venv/`, `mlflow.db`, `mlruns/`, `logs/` or `models/*.pkl`.
