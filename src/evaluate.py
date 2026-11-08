import json

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

METRICS_PATH = BASE_DIR / "models" / "metrics.json"


def evaluate_model():

    if not METRICS_PATH.exists():

        print("Metrics file not found!")

        print("First run train.py")

        return

    with open(METRICS_PATH, "r") as file:

        metrics = json.load(file)

    print("\n========== MODEL EVALUATION ==========\n")

    for name, value in metrics.items():

        print(f"{name.upper()} : {value:.4f}")

    print("\n======================================")


if __name__ == "__main__":

    evaluate_model()