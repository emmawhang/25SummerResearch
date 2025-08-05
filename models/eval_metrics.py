from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.config import Config


def evaluate_model(model, dataset, dataset_name):
    """
    Evaluate model on a given dataset
    Returns dictionary with metrics
    """
    model.eval()
    predictions = []
    true_labels = []

    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE) 

    for batch in loader: 
        inputs = {k: v.to(Config.DEVICE) for k, v in batch.items() if k != 'labels'}
        with torch.no_grad():
            outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "num_samples": len(true_labels)
    }