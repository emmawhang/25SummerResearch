import sys
import os

# LZMA Workaround - Add this BEFORE any other imports
try:
    import lzma
except ImportError:
    # For Python versions where lzma is missing
    from backports import lzma
    sys.modules['lzma'] = lzma
    sys.modules['_lzma'] = lzma

# Ensure the system can find the libraries
os.environ['PATH'] = f"{os.environ.get('PATH', '')}:/usr/local/bin"

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from datasets import load_dataset
from utils.data_loader import prepare_datasets
from utils.eval_metrics import evaluate_model
from config import Config

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def train_model(model, train_data, val_data, epochs, dataset_name):
    train_loader = DataLoader(train_data, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Move batch to device
            inputs = {
                'input_ids': batch['input_ids'].to(Config.DEVICE),
                'attention_mask': batch['attention_mask'].to(Config.DEVICE),
                'labels': batch['labels'].to(Config.DEVICE)
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss/(progress_bar.n+1)})
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, dataset_name)
        print(f"Epoch {epoch+1} - Val Accuracy: {val_metrics['accuracy']:.4f}")
       
def main():
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=4)
    model.to(Config.DEVICE)
    
    # Phase 1: General Training (AG News)
    print("Starting Phase 1: General Training on AG News")
    ag_train, ag_val, ag_test = prepare_datasets("ag_news", tokenizer)
    train_model(model, ag_train, ag_val, Config.AG_NEWS_EPOCHS, "ag_news")
    
    # Evaluate on AG News before domain adaptation
    ag_test_metrics = evaluate_model(model, ag_test, "ag_news")
    
    # Phase 2: Domain Pretraining (PubMedQA)
    print("\nStarting Phase 2: Domain Training on PubMedQA")
    pubmed_train, pubmed_val, pubmed_test = prepare_datasets("pubmedqa", tokenizer)
    train_model(model, pubmed_train, pubmed_val, Config.PUBMEDQA_EPOCHS, "pubmedqa")
    
    # Evaluate on both datasets after domain adaptation
    print("\nFinal Evaluation:")
    ag_test_metrics_post = evaluate_model(model, ag_test, "ag_news")
    pubmed_test_metrics = evaluate_model(model, pubmed_test, "pubmedqa")
    
    # Calculate catastrophic forgetting
    forgetting_score = ag_test_metrics["accuracy"] - ag_test_metrics_post["accuracy"]
    print(f"\nCatastrophic Forgetting Score (ΔAccuracy): {forgetting_score:.4f}")

if __name__ == "__main__":
    main()

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


