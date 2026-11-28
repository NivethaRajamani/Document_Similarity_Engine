"""Similarity search over the document matrix (notebook: 'find_similar_documents')."""
from typing import List, Optional

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from src.config import LABEL_COL, TEXT_COL
from src.embedding import get_document_vector
from src.preprocessing import preprocess_query


class SimilarityEngine:
    """Holds the trained model, corpus and pre-computed document vectors."""

    def __init__(self, corpus: pd.DataFrame, doc_vectors: np.ndarray, model: Word2Vec):
        if len(corpus) != len(doc_vectors):
            raise ValueError("Corpus and document vectors must have the same length.")
        self.corpus = corpus.reset_index(drop=True)
        self.doc_vectors = doc_vectors
        self.model = model

    # ------------------------------------------------------------------ info
    @property
    def clause_types(self) -> List[str]:
        return sorted(self.corpus[LABEL_COL].unique().tolist())

    @property
    def stats(self) -> dict:
        return {
            "documents": len(self.corpus),
            "clause_types": self.corpus[LABEL_COL].nunique(),
            "vocabulary": len(self.model.wv),
            "vector_size": self.model.wv.vector_size,
        }

    # ---------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_n: int = 5,
        clause_types: Optional[List[str]] = None,
        min_score: float = 0.0,
    ) -> pd.DataFrame:
        """Return the top-N most similar clauses for a free-text query.

        With the defaults (no filter, min_score=0) this reproduces the
        notebook's `find_similar_documents` output.
        """
        tokens = preprocess_query(query)
        query_vector = get_document_vector(tokens, self.model).reshape(1, -1)

        similarities = cosine_similarity(query_vector, self.doc_vectors)[0]

        # Optional filter by clause type (UI feature; off by default)
        if clause_types:
            mask = self.corpus[LABEL_COL].isin(clause_types).to_numpy()
            similarities = np.where(mask, similarities, -np.inf)

        top_indices = np.argsort(similarities)[::-1][:top_n]
        top_indices = [i for i in top_indices if np.isfinite(similarities[i])]

        results = self.corpus.iloc[top_indices][[TEXT_COL, LABEL_COL]].copy()
        results["similarity_score"] = similarities[top_indices]
        results = results[results["similarity_score"] >= min_score]
        return results.reset_index(drop=True)

    def similar_words(self, word: str, top_n: int = 10) -> Optional[List[tuple]]:
        """Nearest words in the Word2Vec space, or None if out of vocabulary."""
        word = word.strip().lower()
        if word not in self.model.wv:
            return None
        return self.model.wv.most_similar(word, topn=top_n)

    def query_coverage(self, query: str) -> tuple:
        """(known_tokens, total_tokens) - how much of the query the model understands."""
        tokens = preprocess_query(query)
        known = [t for t in tokens if t in self.model.wv]
        return len(known), len(tokens)
