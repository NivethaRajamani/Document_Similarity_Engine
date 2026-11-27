"""Central configuration: paths and model hyper-parameters.

All values are taken unchanged from the original notebook.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "legal_docs.csv"
MODEL_DIR = BASE_DIR / "models"

WORD2VEC_PATH = MODEL_DIR / "word2vec.model"
DOC_VECTORS_PATH = MODEL_DIR / "document_vectors.npy"
CORPUS_PATH = MODEL_DIR / "corpus.csv"          # cleaned clause_text + clause_type

# ----------------------------------------------------------------------------
# Dataset columns
# ----------------------------------------------------------------------------
TEXT_COL = "clause_text"
LABEL_COL = "clause_type"

# ----------------------------------------------------------------------------
# Tokenisation (gensim.utils.simple_preprocess)
# ----------------------------------------------------------------------------
TOKEN_MIN_LEN = 1
TOKEN_MAX_LEN = 30
TOKEN_DEACC = False

# ----------------------------------------------------------------------------
# Word2Vec hyper-parameters
# ----------------------------------------------------------------------------
W2V_PARAMS = dict(
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,          # skip-gram
    epochs=20,
    seed=42,
)

# ----------------------------------------------------------------------------
# Search defaults
# ----------------------------------------------------------------------------
DEFAULT_TOP_N = 5
MAX_TOP_N = 25
