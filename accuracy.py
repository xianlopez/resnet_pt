import torch


def compute_accuracy(y_pred, y_true):
    # y_pred has shape (batch_size, nclasses), and contains classes scores (logits, not probabilities).
    # y_true has shape (batch_size) and contains the true class indices.
    predictions_classes = torch.argmax(y_pred, dim=1)  # (batch_size)
    comparison = torch.eq(predictions_classes, y_true).float()  # (batch_size)
    accuracy = torch.sum(comparison) / len(comparison)
    return accuracy
