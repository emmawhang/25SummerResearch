# Combating Catastrophic Forgetting via Knowledge-Aware Continual Pretraining

This research project explores how to mitigate catastrophic forgetting during domain adaptation in NLP. We propose a method that combines continual pretraining, regularization techniques (EWC and MAS), and knowledge injection inspired by RAG. The goal is to help lightweight language models like DistilBERT retain general knowledge while adapting to new, domain-specific data.

---

## 🧠 Project Goals

- Apply **continual learning** to NLP domain adaptation
- Evaluate **catastrophic forgetting** on general-domain tasks
- Compare **regularization methods**: EWC and MAS
- Test **knowledge injection** using TF-IDF/FAISS retrieval
- Use **DistilBERT** as a lightweight base model

---

## 📁 Directory Structure

```
25SummerProject/
├── __pycache__/
├── data/
│   ├── ag_news/          # General-domain dataset
│   └── pubmedqa/         # Domain-specific dataset (biomedical QA)
├── models/               
├── utils/
│   ├── __init__.py
│   ├── data_loader.py    # Loads and preprocesses datasets
│   └── eval_metrics.py   # Evaluation utilities
├── continual_learning/   # EWC, MAS, and continual pretraining code
├── config.py             # Hyperparameters and settings
├── main.py               # Training entry point
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📚 Datasets

- **AG News**: General-purpose classification (world, sports, business, tech)
- **PubMedQA**: Biomedical QA reformatted into a 3-class classification problem (yes/no/maybe)

---

## ⚙️ Methods

- **Phase 1: General Training**
  - Fine-tune DistilBERT on AG News
- **Phase 2: Domain Adaptation**
  - Continue training on PubMedQA
  - Use:
    - 🔁 **EWC** (Elastic Weight Consolidation)
    - 📐 **MAS** (Memory Aware Synapses)
    - 🔍 **Knowledge Injection** via retrieval

---

## 🧪 Evaluation

Metrics include:
- Accuracy and perplexity on both domains
- **Catastrophic Forgetting Score** (∆Accuracy on AG News)
- Performance gain with vs. without EWC, MAS, and retrieval

---

## 🚀 How to Run

1. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train on AG News**
   ```bash
   python main.py --phase train_ag
   ```

3. **Continue training on PubMedQA**
   ```bash
   python main.py --phase continual --method baseline|ewc|mas|retrieval
   ```

4. **Evaluate**
   ```bash
   python main.py --phase eval
   ```

---

## 📦 Requirements

- Python 3.8+
- HuggingFace Transformers
- Datasets
- FAISS / Scikit-learn (for retrieval)
- PyTorch
- tqdm

Add all dependencies to `requirements.txt`.

---

## 🧾 Citation

This project is part of a research investigation on continual pretraining and catastrophic forgetting. Please cite this work or contact the author if used in academic contexts.

---

## 📬 Contact

Emma Whang
AI & Statistics @ Purdue University  
✉️ whang@purdue.edu
