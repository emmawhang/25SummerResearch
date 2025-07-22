import torch
import numpy as np
from torch.nn import functional as F
from transformers import Trainer
from typing import Dict, Union, Any

class EWCTrainer(Trainer):
    def __init__(self, ewc_lambda=500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.importance = None
        self.fisher_matrix = None
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Regular forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Add EWC penalty
        if self.fisher_matrix is not None:
            ewc_loss = 0
            for name, param in model.named_parameters():
                if name in self.fisher_matrix:
                    ewc_loss += (self.fisher_matrix[name] * (param - self.importance[name])**2).sum()
            loss += self.ewc_lambda * ewc_loss
        
        return (loss, outputs) if return_outputs else loss

def compute_fisher_matrix(model, dataloader, device):
    fisher_matrix = {}
    importance = {}
    
    # Initialize matrices
    for name, param in model.named_parameters():
        fisher_matrix[name] = torch.zeros_like(param)
        importance[name] = param.data.clone()
    
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        
        # Compute gradients
        model.zero_grad()
        log_probs[:, batch["labels"]].mean().backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_matrix[name] += param.grad.data ** 2
    
    # Average over batches
    for name in fisher_matrix:
        fisher_matrix[name] /= len(dataloader)
    
    return fisher_matrix, importance