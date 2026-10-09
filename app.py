
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("beansense_knn.pkl")
scaler = joblib.load("beansense_scaler.pkl")

DATASET_PATH = "Dry_Bean_Dataset.xlsx"

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/demo")
def demo():
    # Select one real example from the dataset
    data = pd.read_excel(DATASET_PATH)
    sample = data.sample(n=1).iloc[0]

    measurements = {
        feature: float(sample[feature])
        for feature in FEATURES
    }

    # Keep the original label for demonstration comparison
    actual_class = str(sample["Class"])

    return render_template(
        "demo.html",
        measurements=measurements,
        actual_class=actual_class,
        result=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read the 16 measurements in the model's expected order
        values = [
            float(request.form[feature])
            for feature in FEATURES
        ]

        if any(value <= 0 for value in values):
            return "Please enter positive values for all features.", 400

        input_data = pd.DataFrame([values], columns=FEATURES)

        # Apply the scaler fitted during model training
        input_scaled = scaler.transform(input_data)

        # Predict the bean variety
        prediction = model.predict(input_scaled)[0]

        # Neighbor vote proportions; these are not calibrated probabilities
        vote_scores = model.predict_proba(input_scaled)[0]
        class_scores = {
            str(class_name): round(float(score) * 100, 1)
            for class_name, score in zip(model.classes_, vote_scores)
        }

        return render_template(
            "result.html",
            prediction=str(prediction),
            class_scores=class_scores
        )

    except (ValueError, KeyError):
        return "Invalid input. Please check all 16 measurements.", 400


if __name__ == "__main__":
    app.run(debug=True)