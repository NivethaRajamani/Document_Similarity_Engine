"""Loading and cleaning the raw dataset (notebook: 'Data Understanding')."""
from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, LABEL_COL, TEXT_COL


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as provided."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place legal_docs.csv inside the data/ folder."
        )
    return pd.read_csv(path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep required columns, drop nulls / empties / duplicates, reset index."""
    df = df[[TEXT_COL, LABEL_COL]].copy()
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[TEXT_COL].str.strip() != ""]
    df = df.drop_duplicates(subset=[TEXT_COL])
    return df.reset_index(drop=True)


def load_clean_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return clean_dataframe(load_raw_data(path))
