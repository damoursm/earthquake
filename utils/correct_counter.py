

def correct(predictions, labels):
    """
    Counts the number of correct predictions

    Args:
        predictions (torch.Tensor): array of model predictions. Can be either
            1D (batch of class indices) or 2D (batch of prediction vectors)
        labels (torch.Tensor): array of ground truth target labels (1D)
    """
    if predictions.ndim > labels.ndim:
        predictions = predictions.max(1).indices
        # print(predictions)

    correct_predictions = predictions.cpu() == labels.cpu()
    correct_count = correct_predictions.float().sum()
    return correct_count.item()