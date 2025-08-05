from sklearn.feature_extraction.text import TfidfVectorizer
import json
import joblib

def main():
    with open("data/pubmed_abstracts.json") as f:
        corpus = json.load(f)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=30000)
    matrix = vectorizer.fit_transform(corpus)

    joblib.dump(vectorizer, "retriever/tfidf_vectorizer.pkl")
    joblib.dump(matrix, "retriever/tfidf_matrix.pkl")

if __name__ == "__main__":
    main()
