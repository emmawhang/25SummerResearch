from datasets import load_dataset, DatasetDict
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset
import torch
from config import Config

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }
        return item
    
    def __len__(self):
        return len(self.labels)

def prepare_datasets(dataset_name, tokenizer):
    # Load and split dataset
    if dataset_name == "ag_news":
        dataset = load_dataset("ag_news")
        splits = dataset['train'].train_test_split(test_size=0.2, seed=Config.SEED)
        test_valid = splits['test'].train_test_split(test_size=0.5, seed=Config.SEED)
        dataset = DatasetDict({
            'train': splits['train'],
            'validation': test_valid['train'],
            'test': test_valid['test']
        })
    elif dataset_name == "pubmedqa":
        dataset = load_dataset("pubmedqa", "pqa_labeled")
    
    # Tokenization function
    def tokenize_function(examples):
        if dataset_name == "ag_news":
            texts = examples["text"]
        else:
            texts = [f"Question: {q} Context: {c}" for q, c in zip(examples["question"], examples["context"])]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=Config.MAX_LENGTH)
    
    # Apply tokenization
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Prepare torch datasets
    train_dataset = CustomDataset(tokenized_datasets["train"], tokenized_datasets["train"]["label"])
    val_dataset = CustomDataset(tokenized_datasets["validation"], tokenized_datasets["validation"]["label"])
    test_dataset = CustomDataset(tokenized_datasets["test"], tokenized_datasets["test"]["label"])
    
    return train_dataset, val_dataset, test_dataset