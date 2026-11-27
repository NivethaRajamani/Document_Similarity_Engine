"""Text cleaning and tokenisation (notebook: 'clean_text' + simple_preprocess)."""
import re
from typing import List

import pandas as pd
from gensim.utils import simple_preprocess

from src.config import TEXT_COL, TOKEN_DEACC, TOKEN_MAX_LEN, TOKEN_MIN_LEN


def clean_text(text: str) -> str:
    """Lower-case, strip HTML/URLs/punctuation and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)                    # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)                # keep letters/numbers
    text = re.sub(r"\s+", " ", text).strip()                # extra spaces
    return text


def tokenize(cleaned_text: str) -> List[str]:
    """Tokenise already-cleaned text."""
    return simple_preprocess(
        cleaned_text,
        deacc=TOKEN_DEACC,
        min_len=TOKEN_MIN_LEN,
        max_len=TOKEN_MAX_LEN,
    )


def preprocess_query(query: str) -> List[str]:
    """Full pipeline for a single piece of text: clean -> tokenise."""
    return tokenize(clean_text(query))


def add_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Add `cleaned_text` and `tokens` columns to the dataframe."""
    df = df.copy()
    df["cleaned_text"] = df[TEXT_COL].apply(clean_text)
    df["tokens"] = df["cleaned_text"].apply(tokenize)
    return df
