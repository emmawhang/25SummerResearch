import sys
import os
import torch
import numpy as np
from tqdm import tqdm

try:
    import lzma
except ImportError:
    from backports import lzma
    sys.modules['lzma'] = lzma
    sys.modules['_lzma'] = lzma

usr_local_bin = "/usr/local/bin"
current_path = os.environ.get('PATH', '')
if usr_local_bin not in current_path.split(os.pathsep):
    os.environ['PATH'] = f"{current_path}{os.pathsep}{usr_local_bin}" if current_path else usr_local_bin

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_loader import prepare_datasets
from utils.eval_metrics import evaluate_model
from config import Config
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from utils.ewc import EWC
from utils.mas import MAS

def train_model(model, train_data, val_data, epochs, dataset_name):
    num_workers = 2  # Parallel data loading (2-4 for Mac)
    train_loader = DataLoader(
        train_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )

    if Config.DEVICE == "mps":
        print("⚠️ Warning: Mixed precision on MPS is experimental and may cause instability. Falling back to float32.")
        use_mixed_precision = False
    else:
        use_mixed_precision = Config.DEVICE in ['cuda']

    scaler = torch.amp.GradScaler(enabled=use_mixed_precision)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"{dataset_name} Epoch {epoch+1}/{epochs}",
            unit="batch",
        )
        for batch in progress_bar:
            inputs = {
                k: v.to(Config.DEVICE, non_blocking=(Config.DEVICE == "cuda"))
                for k, v in batch.items()
        }
        if Config.DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
                loss = outputs.loss
        else:
            outputs = model(**inputs)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_count += 1
        progress_bar.set_postfix({
            'loss': f"{total_loss/batch_count:.4f}",
            'device': Config.DEVICE
        })

        # Memory cleanup
        if Config.DEVICE == "mps":
            torch.mps.empty_cache()
            
        elif Config.DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Validation (optional, you may want to implement this)
        # val_metrics = evaluate_model(model, val_data, dataset_name)
        # print(f"Epoch {epoch+1} - Val Accuracy: {val_metrics['accuracy']:.4f}")

def train_with_regularization(model, train_data, val_data, epochs, dataset_name, reg_type=None, reg_lambda=0.4, fisher_data=None):
    # This function is similar to train_model, but adds EWC or MAS loss if reg_type is set
    # reg_type: None, 'ewc', or 'mas'
    # fisher_data: data to compute Fisher or MAS importance (usually previous task's data)

    num_workers = 2  # Parallel data loading (2-4 for Mac)
    train_loader = DataLoader(
        train_data,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0)
    )

    if Config.DEVICE == "mps":
        print("⚠️ Warning: Mixed precision on MPS is experimental and may cause instability. Falling back to float32.")
        use_mixed_precision = False

    else:
        use_mixed_precision = Config.DEVICE in ['cuda']

    scaler = torch.amp.GradScaler(enabled=use_mixed_precision)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    if reg_type == 'ewc':
        reg = EWC(model, fisher_data, device=Config.DEVICE)
    elif reg_type == 'mas':
        reg = MAS(model, fisher_data, device=Config.DEVICE)
    else:
        reg = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_count = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"{dataset_name} Epoch {epoch+1}/{epochs}",
            unit="batch",
        )
        for batch in progress_bar:
            inputs = {
                k: v.to(Config.DEVICE, non_blocking=(Config.DEVICE == "cuda"))
                for k, v in batch.items()
        }
        if Config.DEVICE == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**inputs)
                loss = outputs.loss
        else:
            outputs = model(**inputs)
            loss = outputs.loss
            if reg is not None:
                loss += reg_lambda * reg.penalty(model)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_count += 1
        progress_bar.set_postfix({
            'loss': f"{total_loss/batch_count:.4f}",
            'device': Config.DEVICE
        })

        # Memory cleanup
        if Config.DEVICE == "mps":
            torch.mps.empty_cache()
            
        elif Config.DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Validation (optional, you may want to implement this)
        # val_metrics = evaluate_model(model, val_data, dataset_name)
        # print(f"Epoch {epoch+1} - Val Accuracy: {val_metrics['accuracy']:.4f}")

def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=4)
    device = Config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA device not available. Falling back to CPU.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS device not available. Falling back to CPU.")
        device = "cpu"
    model = model.to(device)

    # Prepare datasets
    ag_train, ag_val, ag_test = prepare_datasets("ag_news", tokenizer)
    pubmed_train, pubmed_val, pubmed_test = prepare_datasets(Config.PUBMEDQA_DATASET_NAME, tokenizer)

    # Baseline: Evaluate AG News before PubMedQA training
    ag_test_metrics = evaluate_model(model, ag_test, "AG News")

    # Fine-tune on PubMedQA
    print("\nStarting PubMedQA fine-tuning...")
    train_model(model, pubmed_train, pubmed_val, Config.PUBMEDQA_EPOCHS, "PubMedQA")
    pubmed_test_metrics = evaluate_model(model, pubmed_test, "PubMedQA")

    # Catastrophic Forgetting Score
    ag_test_metrics_post = evaluate_model(model, ag_test, "AG News")
    forgetting_score = ag_test_metrics.get("accuracy", 0.0) - ag_test_metrics_post.get("accuracy", 0.0)
    print(f"\nCatastrophic Forgetting Score (ΔAccuracy): {forgetting_score:.4f}")

    # --- EWC ---
    print("\nStarting EWC regularized training...")
    subset_indices = np.random.choice(len(ag_train), size=1000, replace=False)
    fisher_subset = Subset(ag_train, subset_indices)
    fisher_data = DataLoader(fisher_subset, batch_size=Config.BATCH_SIZE)

    ewc_model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=4).to(device)
    ewc = EWC(ewc_model, fisher_data, device=device)
    train_with_regularization(
        ewc_model, pubmed_train, pubmed_val, Config.PUBMEDQA_EPOCHS,
        "PubMedQA", reg_type='ewc', reg_lambda=0.4, fisher_data=fisher_data
    )
    ewc_ag_test_metrics_post = evaluate_model(ewc_model, ag_test, "AG News (EWC)")
    ewc_pubmed_test_metrics = evaluate_model(ewc_model, pubmed_test, "PubMedQA (EWC)")
    ewc_forgetting_score = ag_test_metrics.get("accuracy", 0.0) - ewc_ag_test_metrics_post.get("accuracy", 0.0)

    # --- MAS ---
    print("\nStarting MAS regularized training...")
    mas_model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=4).to(device)
    mas = MAS(mas_model, fisher_data, device=device)
    train_with_regularization(
        mas_model, pubmed_train, pubmed_val, Config.PUBMEDQA_EPOCHS,
        "PubMedQA", reg_type='mas', reg_lambda=0.4, fisher_data=fisher_data
    )
    mas_ag_test_metrics_post = evaluate_model(mas_model, ag_test, "AG News (MAS)")
    mas_pubmed_test_metrics = evaluate_model(mas_model, pubmed_test, "PubMedQA (MAS)")
    mas_forgetting_score = ag_test_metrics.get("accuracy", 0.0) - mas_ag_test_metrics_post.get("accuracy", 0.0)

    # --- Compare Results ---
    print("\n--- Catastrophic Forgetting Scores ---")
    print(f"Baseline: {forgetting_score:.4f}")
    print(f"EWC:      {ewc_forgetting_score:.4f}")
    print(f"MAS:      {mas_forgetting_score:.4f}")

if __name__ == "__main__":
    main()