import pandas as pd
import numpy as np

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "bank.csv"


def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            "bank.csv not found inside data folder"
        )

    df = pd.read_csv(DATA_PATH, sep=None, engine="python")

    df.columns = df.columns.str.strip()

    df = df.drop_duplicates()

    df = df.replace("unknown", np.nan)

    return df


def preprocess_data():

    df = load_data()

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Values:")
    print(df.duplicated().sum())

    if "y" not in df.columns:

        raise ValueError("Target column 'y' not found")

    # Remove duration to avoid target leakage
    if "duration" in df.columns:

        df = df.drop(columns=["duration"])

    # Separate features and target

    X = df.drop(columns=["y"])

    y = df["y"].astype(str).str.strip().str.lower()

    y = y.map({
        "no": 0,
        "yes": 1
    })

    if y.isnull().any():

        raise ValueError("Unexpected values in target column")

    y = y.astype(int)

    # Identify column types

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print("\nNumerical Columns:")
    print(numerical_columns)

    print("\nCategorical Columns:")
    print(categorical_columns)

    print("\nTarget Distribution:")
    print(y.value_counts())

    return X, y, numerical_columns, categorical_columns


if __name__ == "__main__":

    X, y, numerical, categorical = preprocess_data()

    print("\nPreprocessing Completed Successfully")