"""End-to-end training pipeline and artifact persistence."""
import logging

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from src import config
from src.data_loader import load_clean_data
from src.embedding import build_document_matrix, train_word2vec
from src.preprocessing import add_tokens
from src.similarity import SimilarityEngine

logger = logging.getLogger(__name__)


def artifacts_exist() -> bool:
    return all(
        p.exists()
        for p in (config.WORD2VEC_PATH, config.DOC_VECTORS_PATH, config.CORPUS_PATH)
    )


def train_and_save() -> SimilarityEngine:
    """Load data -> preprocess -> train Word2Vec -> embed docs -> save artifacts."""
    logger.info("Loading and cleaning data ...")
    df = load_clean_data()
    logger.info("Cleaned dataset shape: %s", df.shape)

    logger.info("Cleaning text and tokenising ...")
    df = add_tokens(df)

    logger.info("Training Word2Vec ...")
    model = train_word2vec(df["tokens"].tolist())
    logger.info("Vocabulary size: %d", len(model.wv))

    logger.info("Building document vectors ...")
    doc_vectors = build_document_matrix(df["tokens"].tolist(), model)

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(config.WORD2VEC_PATH))
    np.save(config.DOC_VECTORS_PATH, doc_vectors)
    df[[config.TEXT_COL, config.LABEL_COL]].to_csv(config.CORPUS_PATH, index=False)
    logger.info("Artifacts saved to %s", config.MODEL_DIR)

    return SimilarityEngine(df[[config.TEXT_COL, config.LABEL_COL]], doc_vectors, model)


def load_engine() -> SimilarityEngine:
    """Load a previously trained engine from disk."""
    if not artifacts_exist():
        raise FileNotFoundError("Model artifacts not found. Run `python train.py` first.")
    model = Word2Vec.load(str(config.WORD2VEC_PATH))
    doc_vectors = np.load(config.DOC_VECTORS_PATH)
    corpus = pd.read_csv(config.CORPUS_PATH)
    return SimilarityEngine(corpus, doc_vectors, model)


def get_engine(retrain_if_missing: bool = True) -> SimilarityEngine:
    """Load the engine, training it first if artifacts are missing."""
    if artifacts_exist():
        return load_engine()
    if retrain_if_missing:
        return train_and_save()
    raise FileNotFoundError("Model artifacts not found. Run `python train.py` first.")
