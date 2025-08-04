from utils.retriever import TfidfRetriever

def prepare_augmented_pubmedqa(tokenizer, top_k=5):
    # Load PubMedQA and biomedical corpus
    # corpus = load_biomedical_corpus()
    retriever = TfidfRetriever(corpus)
    # Load PubMedQA dataset
    # dataset = load_pubmedqa()
    augmented_texts = []
    for q, c in zip(dataset["question"], dataset["context"]):
        retrieved = retriever.retrieve(q, top_k=top_k)
        full_context = c + " " + " ".join(retrieved)
        augmented_texts.append(f"Q: {q} C: {full_context}")
    # Tokenize and return as PyTorch Dataset
    encodings = tokenizer(augmented_texts, padding=True, truncation=True, return_tensors="pt")
    dataset = CustomDataset(encodings)
    train_data, val_data, test_data = split_dataset(dataset)
    return train_data, val_data, test_data