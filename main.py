import sys
import os

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

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from utils.data_loader import prepare_datasets
from utils.eval_metrics import evaluate_model
from config import Config

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
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
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(Config.MODEL_NAME)
        model = DistilBertForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=4)
        if Config.DEVICE == "cuda" and not torch.cuda.is_available():
            print("CUDA device not available. Falling back to CPU.")
            device = "cpu"
        elif Config.DEVICE == "mps" and not torch.backends.mps.is_available():
            print("MPS device not available. Falling back to CPU.")
            device = "cpu"
        else:
            device = Config.DEVICE
        model = model.to(device)
    except Exception as e:
        print(f"Error moving model to device '{Config.DEVICE}': {e}")
        print("Falling back to CPU.")
        model = model.to("cpu")

       

    # Phase 1: General Training (AG News) -
    print("Starting Phase 1: General Training on AG News")
    ag_train, ag_val, ag_test = prepare_datasets("ag_news", tokenizer)
    train_model(model, ag_train, ag_val, Config.AG_NEWS_EPOCHS, "AG News")
    ag_test_metrics = evaluate_model(model, ag_test, "AG News")

    # Phase 2: Domain Pretraining (PubMedQA)
    print("\nStarting Phase 2: Domain Training on PubMedQA")
    
    pubmed_train, pubmed_val, pubmed_test = prepare_datasets(Config.PUBMEDQA_DATASET_NAME, tokenizer)
    train_model(model, pubmed_train, pubmed_val, Config.PUBMEDQA_EPOCHS, "PubMedQA")
    pubmed_test_metrics = evaluate_model(model, pubmed_test, "PubMedQA")

    # Catastrophic Forgetting Score
    # You need to evaluate AG News again after PubMedQA training to get ag_test_metrics_post
    ag_test_metrics_post = evaluate_model(model, ag_test, "AG News")
    
    forgetting_score = ag_test_metrics.get("accuracy", 0.0) - ag_test_metrics_post.get("accuracy", 0.0)
    # check 
    print(f"\nCatastrophic Forgetting Score (ΔAccuracy): {forgetting_score:.4f}")

    # --- EWC ---
    print("\nStarting EWC regularized training...")
    # 1. Train on AG News as before
    # 2. Compute Fisher information on AG News
    fisher_data = DataLoader(ag_train, batch_size=Config.BATCH_SIZE)
    ewc_model = DistilBertForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=4).to(Config.DEVICE)
    train_model(ewc_model, ag_train, ag_val, Config.AG_NEWS_EPOCHS, "AG News")
    ewc = EWC(ewc_model, fisher_data, device=Config.DEVICE)
    # 3. Train on PubMedQA with EWC penalty
    train_with_regularization(ewc_model, pubmed_train, pubmed_val, Config.PUBMEDQA_EPOCHS, "PubMedQA", reg_type='ewc', reg_lambda=0.4, fisher_data=fisher_data)
    # 4. Evaluate as before

    # --- MAS ---
    print("\nStarting MAS regularized training...")
    mas_model = DistilBertForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=4).to(Config.DEVICE)
    train_model(mas_model, ag_train, ag_val, Config.AG_NEWS_EPOCHS, "AG News")
    mas = MAS(mas_model, fisher_data, device=Config.DEVICE)
    train_with_regularization(mas_model, pubmed_train, pubmed_val, Config.PUBMEDQA_EPOCHS, "PubMedQA", reg_type='mas', reg_lambda=0.4, fisher_data=fisher_data)
    # 4. Evaluate as before

    # --- Compare Results ---
    # Print or plot the catastrophic forgetting scores for baseline, EWC, and MAS

if __name__ == "__main__":
    main()