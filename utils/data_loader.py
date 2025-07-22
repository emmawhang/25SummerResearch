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
    
def prepare_datasets(dataset_name, tokenizer, train_subset=None, val_subset=None, test_subset=None):

    # Load dataset
    if dataset_name == "qiaojin/PubMedQA":
        dataset = load_dataset(dataset_name, Config.PUBMEDQA_CONFIG)
    else:
        dataset = load_dataset(dataset_name)

    # Optionally select subsets for debugging/speed
    if train_subset:
        dataset["train"] = dataset["train"].select(range(train_subset))
    if val_subset:
        # AG News uses "test" as validation; PubMedQA may use "validation"
        val_split = "test" if "test" in dataset else "validation"
        dataset[val_split] = dataset[val_split].select(range(val_subset))
    if test_subset and "test" in dataset:
        dataset["test"] = dataset["test"].select(range(test_subset))

    # Tokenization function
    def tokenize_function(examples):
        if dataset_name == "ag_news":
            texts = examples["text"]
        else:  # PubMedQA or other datasets
            texts = [f"Q: {q} C: {c}" for q, c in zip(examples["question"], examples["context"])]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=Config.MAX_LENGTH)

    # Tokenize all splits
    tokenized = dataset.map(tokenize_function, batched=True)

    # Choose the correct label column for each dataset
    if dataset_name == "qiaojin/PubMedQA":
        label_col = "final_decision"
    else:
        label_col = "label"

    train_data = CustomDataset(tokenized["train"], tokenized["train"][label_col])
    val_split = "test" if "test" in tokenized else "validation"
    val_data = CustomDataset(tokenized[val_split], tokenized[val_split][label_col])
    test_data = CustomDataset(tokenized["test"], tokenized["test"][label_col]) if "test" in tokenized else val_data

    return train_data, val_data, test_data