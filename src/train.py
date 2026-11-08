import sys

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

import joblib
import json

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from src.preprocessing import preprocess_data


MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


def train_model():

    # Load dataset

    X, y, numerical_columns, categorical_columns = preprocess_data()

    # Train test split

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y

    )

    print("\nTraining Data:", X_train.shape)

    print("Testing Data:", X_test.shape)

    # Numerical pipeline

    numerical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )

    ])

    # Categorical pipeline

    categorical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )

    ])

    # Column Transformer

    preprocessor = ColumnTransformer([

        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )

    ])

    # Random Forest

    rf = RandomForestClassifier(

        random_state=42,
        class_weight="balanced",
        n_jobs=-1

    )

    # Complete ML Pipeline

    pipeline = Pipeline([

        (
            "preprocessing",
            preprocessor
        ),

        (
            "model",
            rf
        )

    ])

    # Stratified K-Fold

    skf = StratifiedKFold(

        n_splits=5,
        shuffle=True,
        random_state=42

    )

    print("\nRunning Cross Validation...")

    scores = cross_val_score(

        pipeline,
        X_train,
        y_train,
        cv=skf,
        scoring="f1",
        n_jobs=-1

    )

    print("\nCross Validation F1 Scores:")

    print(scores)

    print("\nMean CV F1 Score:")

    print(scores.mean())

    # Grid Search

    param_grid = {

        "model__n_estimators": [
            100,
            200
        ],

        "model__max_depth": [
            10,
            20,
            None
        ],

        "model__min_samples_split": [
            2,
            5
        ],

        "model__min_samples_leaf": [
            1,
            2
        ]

    }

    grid_search = GridSearchCV(

        estimator=pipeline,

        param_grid=param_grid,

        cv=skf,

        scoring="f1",

        n_jobs=-1,

        verbose=1

    )

    print("\nStarting GridSearchCV...")

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest Parameters:")

    print(grid_search.best_params_)

    # Predictions

    y_pred = best_model.predict(X_test)

    y_probability = best_model.predict_proba(X_test)[:, 1]

    # Evaluation

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    print("\nModel Evaluation")

    print("Accuracy:", accuracy)

    print("Precision:", precision)

    print("Recall:", recall)

    print("F1 Score:", f1)

    print("ROC AUC:", roc_auc)

    print("\nClassification Report:")

    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")

    print(confusion_matrix(y_test, y_pred))

    # Save Model

    model_path = MODEL_DIR / "model.pkl"

    joblib.dump(best_model, model_path)

    print("\nModel Saved Successfully!")

    print(model_path)

    # Save Metrics

    metrics = {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc": float(roc_auc),

        "cv_f1_score": float(scores.mean())

    }

    metrics_path = MODEL_DIR / "metrics.json"

    with open(metrics_path, "w") as file:

        json.dump(metrics, file, indent=4)

    print("\nTraining Completed Successfully!")


if __name__ == "__main__":

    train_model()