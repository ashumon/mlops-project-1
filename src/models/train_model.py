import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

"""
Trains a Random Forest model on the student placement dataset.
Accepts param for multi-run comparison.
"""
def train(n_estimators: int = 100, random_state: int = 42):
    project_root = Path(__file__).resolve().parents[2]

    # Load dataset
    df = pd.read_csv(project_root / "data" / "raw" / "student_placement_data.csv")

    # Features and target
    X = df.drop("Placement", axis=1)
    y = df["Placement"]

    # Split dataset before training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Keep training and MLflow UI on the same backend store.
    mlflow_db_path = str(project_root / "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

    # Start MLflow experiment
    mlflow.set_experiment("Student Placement Prediction")

    with mlflow.start_run(run_name=f"RF_n{n_estimators}_seed{random_state}"):
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters in MLflow
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        # Log metrics in MLflow
        mlflow.log_metric("accuracy", accuracy)

        # Save model locally
        # models_dir = project_root / "models"
        # models_dir.mkdir(parents=True, exist_ok=True)
        # joblib.dump(model, models_dir / "model.pkl")

        # Log model artifact in MLflow
        mlflow.sklearn.log_model(
            model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"Model trained on student placement dataset! Accuracy: {accuracy}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    print(f"Project root: {project_root}")
    train(n_estimators=100, random_state=42)
    train(n_estimators=200, random_state=42)
    train(n_estimators=150, random_state=0)

    # find best run by highest accuracy across all runs in the experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name("Student Placement Prediction")
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )[0]

    print(f"\nBest run: {best_run.info.run_name} | Accuracy: {best_run.data.metrics['accuracy']:.4f}")

    # save model locally - only the best run's model is saved
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_uri = f"runs:/{best_run.info.run_id}/model"
    best_model = mlflow.sklearn.load_model(best_model_uri)
    
    joblib.dump(best_model, models_dir / "model.pkl")
    print(f"Best model saved to {models_dir}")