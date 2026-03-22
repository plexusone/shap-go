#!/usr/bin/env python3
"""Generate an ONNX model for the Iris classification example.

This script creates a simple logistic regression classifier trained on
the Iris dataset and exports it to ONNX format.

Requirements:
    pip install scikit-learn skl2onnx onnx

Usage:
    python generate_model.py
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def main():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # For binary classification, use versicolor (1) vs not-versicolor (0, 2)
    y_binary = (y == 1).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.4f}")

    # Print model coefficients (useful for LinearSHAP comparison)
    print(f"\nModel coefficients:")
    for i, (name, coef) in enumerate(zip(iris.feature_names, model.coef_[0])):
        print(f"  {name}: {coef:.4f}")
    print(f"  intercept: {model.intercept_[0]:.4f}")

    # Convert to ONNX
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,
        options={id(model): {"zipmap": False}},  # Return probabilities directly
    )

    # Save model
    output_path = "iris_model.onnx"
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"\nONNX model saved to: {output_path}")
    print("\nTo use with SHAP-Go:")
    print("  go run main.go")

    # Print sample predictions for verification
    print("\nSample predictions:")
    samples = [
        ("Setosa", [5.1, 3.5, 1.4, 0.2]),
        ("Versicolor", [6.0, 2.7, 4.5, 1.5]),
        ("Virginica", [7.2, 3.2, 6.0, 1.8]),
    ]
    for name, features in samples:
        proba = model.predict_proba([features])[0]
        print(f"  {name}: P(versicolor)={proba[1]:.4f}")


if __name__ == "__main__":
    main()
