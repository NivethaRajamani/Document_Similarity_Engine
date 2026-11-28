"""CLI: train the Word2Vec model and save all artifacts.

Usage:  python train.py
"""
import logging
import time

from src.pipeline import train_and_save

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    start = time.time()
    engine = train_and_save()
    print("\nTraining complete in %.1fs" % (time.time() - start))
    for key, value in engine.stats.items():
        print(f"  {key:<14}: {value}")
