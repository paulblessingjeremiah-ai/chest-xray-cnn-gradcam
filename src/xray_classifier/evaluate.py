"""Evaluation utilities: confusion matrix and classification report."""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def get_predictions(model, loader, device):
    """Run the model on a loader, return all predictions and true labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path="confusion_matrix.png"):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

    return cm


def print_classification_report(all_labels, all_preds, class_names):
    """Print precision, recall, and F1-score per class."""
    print(classification_report(all_labels, all_preds, target_names=class_names))
