from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    def __init__(self, corpus):
        self.vectorizer = TfidfVectorizer(max_features=10000)
        self.corpus = corpus
        self.corpus_tfidf = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query, top_k=5):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.corpus_tfidf).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]
        return [self.corpus[i] for i in top_indices]