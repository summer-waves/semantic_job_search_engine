import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Embedder:
    _vectorizer = None
    _fitted = False

    def _load(self):
        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                max_features=512,
                stop_words="english",
                ngram_range=(1, 2)
            )

    def fit(self, texts: list[str]):
        self._load()
        self._vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        if not self._fitted:
            raise RuntimeError("Call embedder.fit(texts) first")
        vectors = self._vectorizer.transform(texts).toarray()
        return normalize(vectors).astype("float32")


embedder = Embedder()