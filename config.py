import os
import torch

class Config:
    SEED = 42
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    AG_NEWS_PATH = os.path.join("data", "ag_news") 
    PUBMEDQA_PATH = os.path.join("data", "PubMedQA")
    PUBMEDQA_DATASET_NAME = "qiaojin/PubMedQA"
    PUBMEDQA_CONFIG = "pqa_labeled"  # Config for PubMedQA dataset

    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 512 # reduced from 512 for memory 

    BATCH_SIZE = 1
    LEARNING_RATE = 2e-5
    AG_NEWS_EPOCHS = 1 # training epochs for phase 1 
    PUBMEDQA_EPOCHS = 1 # training epochs for phase 2 