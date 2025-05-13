import pathlib
import joblib
import sys
import yaml
import pandas as pd
from sklearn import metrics
from dvclive import Live
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def evaluate(model, X, y, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        X (pandas.DataFrame): Input data.
        y (pandas.Series): Target column.
        split (str): Dataset name (train/test).
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """
    predictions_by_class = model.predict_proba(X)
    predictions = predictions_by_class[:, 1]

    avg_prec = metrics.average_precision_score(y, predictions)
    roc_auc = metrics.roc_auc_score(y, predictions)

    # Log to dvclive
    live.summary[f"avg_prec_{split}"] = avg_prec
    live.summary[f"roc_auc_{split}"] = roc_auc

    # Log to MLflow
    mlflow.log_metric(f"avg_prec_{split}", avg_prec)
    mlflow.log_metric(f"roc_auc_{split}", roc_auc)

    # Log sklearn plots with dvclive
    live.log_sklearn_plot("roc", y, predictions, name=f"roc/{split}")
    live.log_sklearn_plot("precision_recall", y, predictions, name=f"prc/{split}", drop_intermediate=True)
    live.log_sklearn_plot("confusion_matrix", y, predictions_by_class.argmax(-1), name=f"cm/{split}")

    # Optionally save plots as images and log to MLflow
    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay.from_predictions(y, predictions_by_class.argmax(-1), ax=ax)
    cm_path = f"{save_path}/confusion_matrix_{split}.png"
    fig.savefig(cm_path)
    plt.close(fig)
    mlflow.log_artifact(cm_path)

def save_importance_plot(live, model, feature_names, save_path):
    """
    Save feature importance plot.

    Args:
        live (dvclive.Live): DVCLive instance.
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        feature_names (list): List of feature names.
        save_path (str): Directory to save image file.
    """
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=10)
    forest_importances.plot.bar(ax=ax)
    ax.set_ylabel("Mean decrease in impurity")
    ax.set_title("Top 10 Feature Importances")
    fig.tight_layout()

    importance_path = f"{save_path}/importance.png"
    fig.savefig(importance_path)
    plt.close(fig)

    # Log to DVCLive and MLflow
    live.log_image("importance.png", fig)
    mlflow.log_artifact(importance_path)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent

    model_file = sys.argv[1]  # e.g., "models/model.pkl"
    input_file = sys.argv[2]  # e.g., "/data/processed"

    # Load model
    model = joblib.load(model_file)

    # Load data
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir / "dvclive"
    output_path.mkdir(parents=True, exist_ok=True)

    TARGET = "Class"
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")

    X_train = train_df.drop(columns=TARGET)
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=TARGET)
    y_test = test_df[TARGET]

    feature_names = X_train.columns.tolist()

    # Start MLflow experiment
    mlflow.set_experiment("model_evaluation")

    with mlflow.start_run(run_name="evaluate_rf_model"):
        with Live(output_path.as_posix(), dvcyaml=False) as live:
            # Evaluate both train and test
            evaluate(model, X_train, y_train, "train", live, output_path.as_posix())
            evaluate(model, X_test, y_test, "test", live, output_path.as_posix())

            # Feature importance
            save_importance_plot(live, model, feature_names, output_path.as_posix())

        # Log model as artifact
        signature = infer_signature(X_test, model.predict(X_test))
        mlflow.sklearn.log_model(model, "model", input_example=X_test.head(1), signature=signature)

if __name__ == "__main__":
    main()