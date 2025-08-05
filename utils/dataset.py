from utils.dataset import prepare_augmented_pubmedqa
from retriever.retrieve_top_k import retrieve

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

augmented_dataset = prepare_augmented_pubmedqa(tokenizer, retrieve, top_k=5)
# Split augmented_dataset into train/val/test as needed

class CustomDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def prepare_augmented_pubmedqa(tokenizer, retrieve_fn, top_k=5, path="data/pubmedqa/train.csv"):
    import pandas as pd
    df = pd.read_csv(path)
    # Binary classification: filter out "maybe"
    df = df[df["label"].isin(["yes", "no"])]
    label_map = {"yes": 0, "no": 1}
    questions = df["question"].tolist()
    contexts = df["context"].fillna("").tolist()
    labels = [label_map[l] for l in df["label"]]

    augmented_texts = []
    for q, c in zip(questions, contexts):

        retrieved = retrieve_fn(q, k=top_k)
        full_context = c + " " + " ".join(retrieved)
        augmented_texts.append(f"Q: {q} C: {full_context}")

    encodings = tokenizer(augmented_texts, padding=True, truncation=True, return_tensors="pt")
    dataset = CustomDataset(encodings, labels)
    # You can split into train/val/test as needed
    return dataset