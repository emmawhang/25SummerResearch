from datasets import load_dataset
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
        val_split = "validation" if "validation" in dataset else "test"
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
        label_map = {"yes": 0, "no": 1, "maybe": 2}
        def map_labels(labels):
            return [label_map[l] for l in labels]
        train_labels = map_labels(tokenized["train"][label_col])
        if "validation" in tokenized:
            val_labels = map_labels(tokenized["validation"][label_col])
        elif "test" in tokenized:
            val_labels = map_labels(tokenized["test"][label_col])
        else:
            val_labels = train_labels
        if "test" in tokenized:
            test_labels = map_labels(tokenized["test"][label_col])
        elif "validation" in tokenized:
            test_labels = map_labels(tokenized["validation"][label_col])
        else:
            test_labels = val_labels
    else:
        label_col = "label"
        train_labels = tokenized["train"][label_col]
        if "validation" in tokenized:
            val_labels = tokenized["validation"][label_col]
        elif "test" in tokenized:
            val_labels = tokenized["test"][label_col]
        else:
            val_labels = train_labels
        if "test" in tokenized:
            test_labels = tokenized["test"][label_col]
        elif "validation" in tokenized:
            test_labels = tokenized["validation"][label_col]
        else:
            test_labels = val_labels

    train_data = CustomDataset(tokenized["train"], train_labels)
    val_data = CustomDataset(tokenized["validation"] if "validation" in tokenized else tokenized["test"] if "test" in tokenized else tokenized["train"], val_labels)
    test_data = CustomDataset(tokenized["test"] if "test" in tokenized else tokenized["validation"] if "validation" in tokenized else tokenized["train"], test_labels)

    return train_data, val_data, test_data
