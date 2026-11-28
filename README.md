# Legal Clause Similarity Engine

Find similar legal clauses using **Word2Vec** (skip-gram) document embeddings and **cosine similarity**.
The logic is identical to `Document_Similarity_Engine.ipynb`, split into modules, with a Streamlit UI.

## Project structure
```
document_similarity_engine/
├── app.py               # Streamlit UI
├── train.py             # Train model + save artifacts
├── search_cli.py        # Terminal search
├── requirements.txt
├── data/legal_docs.csv  # dataset
├── models/              # generated: word2vec.model, document_vectors.npy, corpus.csv
├── src/
│   ├── config.py        # paths + hyper-parameters (same as notebook)
│   ├── data_loader.py   # load + clean dataframe
│   ├── preprocessing.py # clean_text + tokenisation
│   ├── embedding.py     # Word2Vec training + averaged document vectors
│   ├── similarity.py    # SimilarityEngine (cosine search, word neighbours)
│   └── pipeline.py      # train_and_save / load_engine / get_engine
└── tests/test_engine.py
```

## Setup (VS Code terminal)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run
```bash
python train.py                 # optional: ~2 min, creates models/ (the app trains automatically if missing)
streamlit run app.py            # opens the UI at http://localhost:8501
python search_cli.py "Either party may terminate this agreement by providing written notice." -n 5
pytest -q                       # smoke tests (needs trained model)
```

## UI features
- Paste any clause, get top-N similar clauses with score bars and clause-type badges
- Sidebar: number of results, minimum similarity, filter by clause type
- Example queries, query vocabulary coverage, CSV download of results
- Word explorer tab (nearest words in the Word2Vec space)
