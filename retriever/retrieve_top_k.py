import joblib
import json
import numpy as np

def retrieve(question, k=3):
    vectorizer = joblib.load("retriever/tfidf_vectorizer.pkl")
    matrix = joblib.load("retriever/tfidf_matrix.pkl")

    with open("data/pubmed_abstracts.json") as f:
        corpus = json.load(f)

    question_vec = vectorizer.transform([question])
    sim = matrix @ question_vec.T
    sim = sim.toarray().flatten()

    top_k = np.argsort(sim)[::-1][:k]
    return [corpus[i] for i in top_k]

def main():
    question = "Does aspirin help with heart disease?"
    top_k = retrieve(question)
    print("\n".join(top_k))

if __name__ == "__main__":
    main()
