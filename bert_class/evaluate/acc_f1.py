import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report


def evaluate(y_pred, y_true):
    _, y_pred = torch.max(y_pred.data, 1)
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    f1 = f1_score(y_true, y_pred, average="macro")
    correct = np.sum((y_true == y_pred).astype(int))
    acc = correct / y_pred.shape[0]
    return (acc, f1)


def class_report(y_pred, y_true):
    _, y_pred = torch.max(y_pred.data, 1)
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    classify_report = classification_report(y_true, y_pred)
    print('\n\nclassify_report:\n', classify_report)