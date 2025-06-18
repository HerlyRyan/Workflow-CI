import os
import warnings
import pandas as pd
import mlflow
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

def setup_mlflow():
    if os.getenv("ENV") != "production":
        load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/HerlyRyan/ekstrovert-introvert-behavior.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Ekstrovert Introvert Modelling")

    print(f"✅ MLflow tracking URI set to: {tracking_uri}")

def download_model(run_id, artifact_path, output_dir_name):
    client = mlflow.tracking.MlflowClient()
    output_dir_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir_name)
    os.makedirs(output_dir_name, exist_ok=True)
    local_path = client.download_artifacts(run_id, artifact_path, output_dir_name)
    print(f"📥 Model downloaded to: {local_path}")

def run_model(X_train, y_train, X_test, y_test, run_id):
    input_example = pd.DataFrame(X_train).head(5)
    n_estimators = 257
    max_depth = 1
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = model.score(X_test, y_test)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted", recall)
    mlflow.log_metric("f1_weighted", f1)

    mlflow.sklearn.log_model(model, "modelling", input_example=input_example)
    print(f"✅ Model logged. accuracy={accuracy:.4f}, precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}")
    download_model(run_id, "modelling", "output")

def main():
    warnings.filterwarnings("ignore")
    setup_mlflow()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessing", "output")
    X_train = pd.read_csv(os.path.join(PREPROCESS_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PREPROCESS_DIR, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(PREPROCESS_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PREPROCESS_DIR, "y_test.csv"))

    if mlflow.active_run() is None:
        with mlflow.start_run() as run:
            run_model(X_train, y_train, X_test, y_test, run.info.run_id)
    else:
        run = mlflow.active_run()
        run_model(X_train, y_train, X_test, y_test, run.info.run_id)

if __name__ == "__main__":
    main()
