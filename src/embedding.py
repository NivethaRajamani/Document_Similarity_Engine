"""Word2Vec training and document embedding (notebook: 'Model Building' + 'Document embedding')."""
from typing import Iterable, List

import numpy as np
from gensim.models import Word2Vec

from src.config import W2V_PARAMS


def train_word2vec(sentences: List[List[str]]) -> Word2Vec:
    """Train a skip-gram Word2Vec model with the notebook's hyper-parameters."""
    return Word2Vec(sentences=sentences, **W2V_PARAMS)


def get_document_vector(tokens: Iterable[str], model: Word2Vec) -> np.ndarray:
    """Average of in-vocabulary word vectors; zero vector if none are known."""
    word_vectors = [model.wv[w] for w in tokens if w in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.wv.vector_size)
    return np.mean(word_vectors, axis=0)


def build_document_matrix(token_lists: List[List[str]], model: Word2Vec) -> np.ndarray:
    """Stack document vectors into an (n_docs, vector_size) matrix."""
    return np.vstack([get_document_vector(t, model) for t in token_lists])
