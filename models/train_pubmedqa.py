from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.data_loader import prepare_datasets
from utils.train import train_model

def train_pubmedqa(model_name, epochs, batch_size, augmented=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if augmented:
        from utils.data_loader import prepare_augmented_pubmedqa
        train_data, val_data, test_data = prepare_augmented_pubmedqa(tokenizer)
    else:
        train_data, val_data, test_data = prepare_datasets("qiaojin/PubMedQA", tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    train_model(model, train_data, val_data, epochs, "PubMedQA")
    return model, test_data