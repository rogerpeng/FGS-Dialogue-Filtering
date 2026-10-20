"""Interchangeable text classifiers. All fitted transforms see training data only."""
import copy
import hashlib
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def stable_hash(word):
    return int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16)


class TextClassifier:
    def __init__(self, kind="tfidf", seed=42, learning_rate=0.001, C=1.0,
                 epochs=300, patience=25, embedding_model=None, embedding_revision=None,
                 local_files_only=False, cache_dir=None):
        if kind not in {"tfidf", "word2vec", "word2vec_linear", "transformer", "transformer_mlp"}:
            raise ValueError(f"Unknown classifier: {kind}")
        self.kind, self.seed, self.learning_rate, self.C = kind, seed, learning_rate, C
        self.feature_kind = "word2vec" if kind.startswith("word2vec") else "transformer" if kind.startswith("transformer") else "tfidf"
        self.neural = kind in {"word2vec", "transformer_mlp"}
        self.epochs, self.patience = epochs, patience
        self.embedding_model, self.embedding_revision = embedding_model, embedding_revision
        self.local_files_only, self.cache_dir = local_files_only, cache_dir
        self.threshold, self.history, self.best_epoch = 0.5, [], None
        self.encoder = None

    def _tokens(self, texts):
        import jieba
        # Keep tokenizer cache in the experiment folder instead of system temp.
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            jieba.dt.tmp_dir = str(Path(self.cache_dir).resolve())
        return [[w for w in jieba.lcut(t) if w.strip()] for t in texts]

    def _encoder(self):
        if self.encoder is None:
            if not self.embedding_model:
                raise ValueError("Transformer baseline requires an explicit embedding_model")
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.embedding_model, revision=self.embedding_revision,
                                               local_files_only=self.local_files_only,
                                               cache_folder=self.cache_dir, trust_remote_code=False)
        return self.encoder

    def _features(self, texts, fit=False):
        if self.feature_kind == "tfidf":
            if fit:
                self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1,
                                                  max_features=30000, sublinear_tf=True)
                return self.vectorizer.fit_transform(texts)
            return self.vectorizer.transform(texts)
        if self.feature_kind == "word2vec":
            tokens = self._tokens(texts)
            if fit:
                from gensim.models import Word2Vec
                self.word_vectors = Word2Vec(sentences=tokens, vector_size=100, window=5,
                                            min_count=2, workers=1, sg=1, seed=self.seed,
                                            epochs=20, hashfxn=stable_hash).wv
            raw = np.asarray([np.mean([self.word_vectors[w] for w in ws if w in self.word_vectors], axis=0)
                              if any(w in self.word_vectors for w in ws) else np.zeros(100)
                              for ws in tokens], dtype=np.float32)
        else:
            raw = self._encoder().encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                         show_progress_bar=False, batch_size=32)
        if fit:
            self.scaler = StandardScaler().fit(raw)
        return self.scaler.transform(raw)

    def fit(self, train_texts, train_labels, validation_texts, validation_labels):
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")
        X = self._features(train_texts, fit=True)
        V = self._features(validation_texts)
        y, vy = np.asarray(train_labels), np.asarray(validation_labels)
        if not self.neural:
            self.classifier = LogisticRegression(C=self.C, max_iter=2000, random_state=self.seed)
            self.classifier.fit(X, y)
            return self
        model = MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), activation="relu", solver="adam",
                              learning_rate_init=self.learning_rate, batch_size=min(16, len(y)),
                              alpha=0.0, random_state=self.seed, shuffle=True)
        best, stale, best_loss = None, 0, float("inf")
        for epoch in range(1, self.epochs + 1):
            model.partial_fit(X, y, classes=np.array([0, 1]))
            vp, tp = model.predict_proba(V), model.predict_proba(X)
            loss = float(log_loss(vy, vp, labels=[0, 1]))
            self.history.append({"epoch": epoch, "train_loss": float(log_loss(y, tp, labels=[0, 1])),
                                 "validation_loss": loss,
                                 "train_accuracy": float(accuracy_score(y, tp.argmax(axis=1))),
                                 "validation_accuracy": float(accuracy_score(vy, vp.argmax(axis=1)))})
            if loss < best_loss - 1e-7:
                best_loss, best, stale, self.best_epoch = loss, copy.deepcopy(model), 0, epoch
            else:
                stale += 1
                if stale >= self.patience:
                    break
        # Inference MUST use the best validation checkpoint, not the final epoch.
        self.classifier = best
        return self

    def probabilities(self, texts):
        return self.classifier.predict_proba(self._features(texts))[:, list(self.classifier.classes_).index(1)]

    def accepts(self, text):
        return bool(self.probabilities([text])[0] >= self.threshold)

    def oov_rate(self, texts):
        if self.feature_kind != "word2vec":
            return None
        words = [w for ws in self._tokens(texts) for w in ws]
        return sum(w not in self.word_vectors for w in words) / len(words) if words else 1.0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["encoder"] = None  # external model is reloaded by pinned id/revision
        return state
