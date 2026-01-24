# src/models.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from typing import Dict, Optional, Tuple


class MLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def train_and_evaluate_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    total_num_classes: int,
    seed: int = 42,
    batch_size: int = 32,
    lr: float = 0.01,
    momentum: float = 0.5,
    max_epochs: int = 50,
    device: Optional[str] = None,
) -> Tuple[Dict[str, float], nn.Module]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    input_size = X_train.shape[1]
    model = MLP(input_size=input_size, num_classes=total_num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    model.train()
    for _ in range(max_epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    metrics: Dict[str, float] = {"Accuracy": float("nan")}
    if X_test.size != 0 and y_test.size != 0 and len(y_test) > 0:
        acc = evaluate_pytorch_model(model, X_test, y_test, batch_size=batch_size, device=device)
        metrics["Accuracy"] = float(acc)

    return metrics, model


def evaluate_pytorch_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> float:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)

    if y_test.size == 0 or len(y_test) == 0 or X_test.size == 0:
        return float("nan")

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return float(accuracy_score(all_labels, all_preds))
