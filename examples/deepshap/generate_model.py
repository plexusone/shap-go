#!/usr/bin/env python3
"""Generate an ONNX MLP model for the DeepSHAP example.

This script creates a multi-layer perceptron (MLP) neural network trained
on the Iris dataset and exports it to ONNX format with named intermediate
outputs for DeepSHAP activation capture.

Requirements:
    pip install scikit-learn skl2onnx onnx torch

Usage:
    python generate_model.py
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class IrisMLP(nn.Module):
    """Simple MLP for Iris classification."""

    def __init__(self, input_size=4, hidden_size=8, output_size=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


def main():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Binary classification: versicolor (1) vs not-versicolor (0, 2)
    y_binary = (y == 1).astype(np.float32)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42
    )

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).unsqueeze(1)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test).unsqueeze(1)

    # Create and train model
    model = IrisMLP()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training MLP...")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        predictions = (test_outputs > 0.5).float()
        accuracy = (predictions == y_test_t).float().mean()
        print(f"\nTest accuracy: {accuracy.item():.4f}")

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 4)

    output_path = "mlp_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=12,
    )

    print(f"\nONNX model saved to: {output_path}")

    # Print layer info
    print("\nModel architecture:")
    for name, module in model.named_modules():
        if name:
            print(f"  {name}: {module}")

    # Print sample predictions (using original scale for readability)
    print("\nSample predictions (using scaled features):")
    samples = [
        ("Setosa", [5.1, 3.5, 1.4, 0.2]),
        ("Versicolor", [6.0, 2.7, 4.5, 1.5]),
        ("Virginica", [7.2, 3.2, 6.0, 1.8]),
    ]

    for name, features in samples:
        scaled = scaler.transform([features])
        with torch.no_grad():
            pred = model(torch.from_numpy(scaled.astype(np.float32)))
        print(f"  {name}: P(versicolor)={pred.item():.4f}")

    # Note about scaling
    print("\nNote: The Go example uses unscaled Iris features directly.")
    print("For production use, apply the same scaling as training.")
    print(f"\nScaler mean: {scaler.mean_}")
    print(f"Scaler std:  {scaler.scale_}")


if __name__ == "__main__":
    main()
