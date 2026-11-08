import joblib
import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "model.pkl"


FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]


def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Model not found. First run src/train.py"
        )

    return joblib.load(MODEL_PATH)


def predict_customer(input_data):

    model = load_model()

    input_df = pd.DataFrame([input_data])

    input_df = input_df[FEATURES]

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    result = {

        "prediction": int(prediction),

        "probability": float(probability),

        "status": (
            "Customer Likely to Subscribe"
            if prediction == 1
            else "Customer Unlikely to Subscribe"
        )

    }

    return result


def predict_batch(input_df):

    model = load_model()

    missing_columns = [
        col for col in FEATURES
        if col not in input_df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {', '.join(missing_columns)}"
        )

    input_df = input_df[FEATURES].copy()

    predictions = model.predict(input_df)

    probabilities = model.predict_proba(input_df)[:, 1]

    input_df["prediction"] = predictions

    input_df["probability"] = probabilities

    input_df["status"] = [
        "Customer Likely to Subscribe"
        if prediction == 1
        else "Customer Unlikely to Subscribe"
        for prediction in predictions
    ]

    return input_df


if __name__ == "__main__":

    sample = {

        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"

    }

    print(predict_customer(sample))