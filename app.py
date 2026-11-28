"""Streamlit UI for the Document Similarity Engine.

Run:  streamlit run app.py
"""
import html

import pandas as pd
import streamlit as st

from src import config
from src.pipeline import artifacts_exist, get_engine, train_and_save

st.set_page_config(
    page_title="Legal Clause Similarity Engine",
    page_icon="⚖️",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1100px;}
    .hero h1 {margin-bottom: 0;}
    .hero p {color: #6b7280; margin-top: .25rem;}
    .result-card {
        border: 1px solid rgba(128,128,128,.3); border-radius: 12px;
        padding: 1rem 1.2rem; margin-bottom: .9rem;
        background: rgba(128,128,128,.06);
    }
    .result-head {display:flex; justify-content:space-between; align-items:center; margin-bottom:.5rem;}
    .rank {font-weight:700; opacity:.7;}
    .badge {
        background:#4f46e5; color:#fff; padding:2px 10px; border-radius:999px;
        font-size:.78rem; font-weight:600;
    }
    .score {font-variant-numeric: tabular-nums; font-weight:700;}
    .bar-bg {background: rgba(128,128,128,.25); border-radius: 6px; height: 8px; margin: .35rem 0 .8rem;}
    .bar-fg {height: 8px; border-radius: 6px; background: linear-gradient(90deg,#6366f1,#22c55e);}
    .clause {line-height: 1.55; font-size: .95rem; white-space: pre-wrap;}
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = {
    "Asset transfer restriction": "The borrower shall not transfer its assets without prior written consent from the lender.",
    "Payment terms": "The company shall make all payments within thirty days of receiving the invoice.",
    "Termination": "Either party may terminate this agreement by providing written notice.",
    "Confidentiality": "The receiving party shall keep all confidential information strictly confidential.",
    "Governing law": "This agreement shall be governed by and construed in accordance with the laws of the State of New York.",
}


# ----------------------------------------------------------------------------
# Engine loading (cached across reruns)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_engine_cached():
    if artifacts_exist():
        return get_engine(retrain_if_missing=False)
    return train_and_save()


if not artifacts_exist():
    st.info(
        "First run: training the Word2Vec model on the dataset. "
        "This takes a couple of minutes and only happens once."
    )
with st.spinner("Loading model ..."):
    try:
        engine = load_engine_cached()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Search settings")
    top_n = st.slider("Number of results", 1, config.MAX_TOP_N, config.DEFAULT_TOP_N)
    min_score = st.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.01)
    selected_types = st.multiselect(
        "Filter by clause type",
        options=engine.clause_types,
        placeholder="All clause types",
    )

    st.divider()
    st.subheader("Corpus")
    s = engine.stats
    c1, c2 = st.columns(2)
    c1.metric("Clauses", f"{s['documents']:,}")
    c2.metric("Types", s["clause_types"])
    c1.metric("Vocabulary", f"{s['vocabulary']:,}")
    c2.metric("Vector dim", s["vector_size"])


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def render_result(rank: int, row: pd.Series) -> None:
    score = float(row["similarity_score"])
    width = max(0.0, min(score, 1.0)) * 100
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-head">
            <span class="rank">#{rank}</span>
            <span><span class="badge">{html.escape(str(row[config.LABEL_COL]))}</span>
            &nbsp; <span class="score">{score:.4f}</span></span>
          </div>
          <div class="bar-bg"><div class="bar-fg" style="width:{width:.1f}%"></div></div>
          <div class="clause">{html.escape(str(row[config.TEXT_COL]).strip())}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_example() -> None:
    st.session_state["query_text"] = EXAMPLES[st.session_state["example_choice"]]


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>⚖️ Legal Clause Similarity Engine</h1>'
    "<p>Paste a clause and find the most similar clauses in the corpus "
    "(Word2Vec embeddings + cosine similarity).</p></div>",
    unsafe_allow_html=True,
)

tab_search, tab_words, tab_about = st.tabs(["🔍 Find similar clauses", "🔤 Word explorer", "ℹ️ About"])

# ----------------------------------------------------------------------------
# Tab 1 - clause search
# ----------------------------------------------------------------------------
with tab_search:
    ex_col, btn_col = st.columns([4, 1])
    ex_col.selectbox("Try an example", list(EXAMPLES), key="example_choice", label_visibility="collapsed")
    btn_col.button("Load example", on_click=load_example, width="stretch")

    query = st.text_area(
        "Your clause or text",
        key="query_text",
        height=140,
        placeholder="e.g. Either party may terminate this agreement by providing written notice.",
    )
    search_clicked = st.button("Find similar clauses", type="primary")

    if search_clicked or st.session_state.get("last_query") == query and query.strip():
        if not query.strip():
            st.warning("Please enter some text to search.")
        else:
            st.session_state["last_query"] = query
            known, total = engine.query_coverage(query)
            if known == 0:
                st.warning(
                    "None of the words in your text are in the model's vocabulary, "
                    "so no meaningful match can be found. Try legal wording."
                )
            else:
                results = engine.search(
                    query, top_n=top_n, clause_types=selected_types or None, min_score=min_score
                )
                st.caption(f"Query understood: {known} of {total} words are in the vocabulary.")
                if results.empty:
                    st.info("No clauses matched your filters. Lower the minimum similarity or clear the filter.")
                else:
                    top = results.iloc[0]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Results", len(results))
                    m2.metric("Best score", f"{top['similarity_score']:.4f}")
                    m3.metric("Top clause type", top[config.LABEL_COL])

                    for i, row in results.iterrows():
                        render_result(i + 1, row)

                    st.download_button(
                        "⬇️ Download results (CSV)",
                        results.to_csv(index=False).encode("utf-8"),
                        file_name="similar_clauses.csv",
                        mime="text/csv",
                    )

# ----------------------------------------------------------------------------
# Tab 2 - word explorer
# ----------------------------------------------------------------------------
with tab_words:
    st.write("Explore what the Word2Vec model learned: type a word to see its nearest neighbours.")
    wc1, wc2 = st.columns([3, 1])
    word = wc1.text_input("Word", value="borrower", placeholder="e.g. payment, interest, lender")
    n_words = wc2.number_input("Neighbours", 3, 30, 10)
    if word.strip():
        neighbours = engine.similar_words(word, int(n_words))
        if neighbours is None:
            st.warning(f"'{word}' is not in the vocabulary.")
        else:
            df_words = pd.DataFrame(neighbours, columns=["word", "similarity"])
            st.dataframe(
                df_words,
                width="stretch",
                hide_index=True,
                column_config={
                    "similarity": st.column_config.ProgressColumn(
                        "similarity", min_value=0.0, max_value=1.0, format="%.4f"
                    )
                },
            )

# ----------------------------------------------------------------------------
# Tab 3 - about
# ----------------------------------------------------------------------------
with tab_about:
    st.markdown(
        f"""
**How it works**

1. **Clean** each clause (lower-case, remove HTML/URLs/punctuation) and tokenise it.
2. **Train** a skip-gram Word2Vec model (`vector_size={config.W2V_PARAMS['vector_size']}`,
   `window={config.W2V_PARAMS['window']}`, `min_count={config.W2V_PARAMS['min_count']}`,
   `epochs={config.W2V_PARAMS['epochs']}`).
3. **Embed** every clause as the average of its word vectors.
4. **Search**: your text goes through the same pipeline and is compared to all clauses
   with cosine similarity; the top matches are returned.

**Note:** Averaged Word2Vec captures topical/semantic closeness, not exact wording.
Scores are relative; compare them across results rather than reading them as probabilities.
        """
    )
