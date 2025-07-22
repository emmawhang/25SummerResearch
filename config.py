import os
import torch

class Config:
    SEED = 42
    DEVICE = "cpu"
    AG_NEWS_PATH = os.path.join("data", "ag_news")
    PUBMEDQA_PATH = os.path.join("data", "pubmedqa")
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 128
    BATCH_SIZE = 4 if DEVICE == "mps" else 16
    LEARNING_RATE = 2e-5
    AG_NEWS_EPOCHS = 1
    PUBMEDQA_EPOCHS = 5