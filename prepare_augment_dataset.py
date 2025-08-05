from utils.retriever import TfidfRetriever
from utils.dataset import CustomDataset, split_dataset  # You must define these
from utils.data_loader import load_pubmedqa, load_biomedical_corpus  # Add these helpers
from transformers import PreTrainedTokenizer

def prepare_augmented_pubmedqa(tokenizer: PreTrainedTokenizer, top_k=5):
    # Step 1: Load biomedical corpus
    corpus = load_biomedical_corpus()  # returns List[str]
    retriever = TfidfRetriever(corpus)

    # Step 2: Load PubMedQA
    dataset = load_pubmedqa()  # returns dict with keys: question, context, label

    # Step 3: Augment inputs
    augmented_texts = []
    labels = []

    for q, c, label in zip(dataset["question"], dataset["context"], dataset["label"]):
        retrieved = retriever.retrieve(q, top_k=top_k)
        full_context = c + " " + " ".join(retrieved)
        input_text = f"Q: {q} C: {full_context}"
        augmented_texts.append(input_text)
        labels.append(label)

    # Step 4: Tokenize
    encodings = tokenizer(augmented_texts, padding=True, truncation=True, return_tensors="pt")

    # Step 5: Create dataset
    dataset = CustomDataset(encodings, labels)
    train_data, val_data, test_data = split_dataset(dataset)

    return train_data, val_data, test_data
