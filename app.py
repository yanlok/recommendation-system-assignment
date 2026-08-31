"""Mood-Based Movie Recommendation System (Streamlit UI).

UI only: user/mood selection, the recommend button and result display.
All recommendation logic lives in src/ (see documentation plan §34).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import collaborative, content_based, data_loader, hybrid, mood_filter, utils

st.set_page_config(page_title="Mood Based Recommendation System", page_icon="🎬", layout="wide")


# ── Cached loading (plan §22, §23) ────────────────────────────────
@st.cache_resource(show_spinner="Loading recommendation models...")
def load_everything():
    artifacts = data_loader.load_artifacts()
    lookup = data_loader.load_movie_lookup()
    ratings = data_loader.load_ratings_small()
    models = {
        "Collaborative Filtering": collaborative.CollaborativeFilter(
            artifacts["user_item"], artifacts["user_sim_cf"],
            artifacts["user_ids"], artifacts["movie_ids"]),
        "Content-Based Filtering": content_based.ContentBasedFilter(
            artifacts["user_item"], artifacts["item_sim_content"],
            artifacts["user_ids"], artifacts["movie_ids"]),
        "Hybrid": hybrid.HybridRecommender(
            artifacts["user_item"], artifacts["user_sim_raw"],
            artifacts["item_sim_hybrid"], artifacts["user_ids"],
            artifacts["movie_ids"], artifacts["global_mean"]),
    }
    rating_counts = ratings.groupby("userId")["rating"].size()
    return lookup, ratings, models, rating_counts


@st.cache_data(show_spinner=False)
def popularity_ranking() -> pd.Series:
    lookup, *_ = load_everything()
    return utils.popularity_score(lookup).sort_values(ascending=False)


LOOKUP, RATINGS, MODELS, RATING_COUNTS = load_everything()
POPULARITY = popularity_ranking()
MOOD_GENRES = mood_filter.load_mood_genres()

ALGORITHM_KEYS = {
    "Collaborative Filtering": "collaborative",
    "Content-Based Filtering": "content",
    "Hybrid": "hybrid",
}


# ── Core recommendation pipeline (matches the activity diagram) ──
def generate_recommendations(user_id, mood, n, method):
    """Mood filter -> remove already-rated -> algorithm scores -> rank -> Top-N."""
    rated = set(RATINGS.loc[RATINGS["userId"] == user_id, "ratingId"])
    candidates = [m for m in mood_filter.mood_candidate_ids(mood, LOOKUP) if m not in rated]

    if not candidates:
        return pd.DataFrame(), {"empty": True}

    user_is_sparse = RATING_COUNTS.get(user_id, 0) < utils.SPARSE_USER_THRESHOLD

    if user_is_sparse:
        # Cold-start fallback (plan §19): mood match + popularity
        ranked = POPULARITY.reindex(candidates).dropna().sort_values(ascending=False)
        fillers = set(ranked.index)
        notice = ("This user has limited rating history, so recommendations are "
                  "based mainly on mood and movie popularity.")
    else:
        scores = MODELS[method].score_candidates(user_id, candidates)
        ranked = scores.dropna().sort_values(ascending=False)
        fillers = set()
        notice = None
        if len(ranked) < n:
            # Candidate starvation: fill remaining slots by popularity
            remaining = [c for c in candidates if c not in ranked.index]
            extra = (POPULARITY.reindex(remaining).dropna()
                     .sort_values(ascending=False).head(n - len(ranked)))
            fillers = set(extra.index)
            ranked = pd.concat([ranked, extra])

    top = ranked.head(n)
    result = top.rename("score").reset_index().rename(columns={"index": "ratingId"})
    result = result.merge(LOOKUP, on="ratingId", how="left")
    result["rank"] = range(1, len(result) + 1)
    result["source"] = ["popularity" if m in fillers else "model" for m in result["ratingId"]]
    return result, {"empty": False, "n_candidates": len(candidates), "notice": notice}


def render_result_cards(result, mood, method):
    method_key = ALGORITHM_KEYS.get(method, "popularity")
    st.caption(f"All picks match your {mood} mood ({', '.join(MOOD_GENRES[mood])}).")
    for _, movie in result.iterrows():
        with st.container(border=True):
            title_col, score_col = st.columns([5, 1])
            title_col.markdown(f"**#{movie['rank']} · {movie['title']}**")
            score_col.markdown(f"⭐ **{movie['score']:.2f} / 5**")
            reason = (utils.reason_for("popularity") if movie["source"] == "popularity"
                      else utils.reason_for(method_key))
            st.caption(f"{str(movie['genres_str']).replace('|', ' • ')} — {reason}")


def user_selector(key, label="Select User"):
    demo_labels = {f"{name} — {desc}": uid for name, uid, desc in utils.DEMO_USERS}
    choice = st.selectbox(label, list(demo_labels), index=0, key=key)
    return demo_labels[choice]


def mood_selector(key, label="How are you feeling today?"):
    icons = {"Happy": "😀", "Sad": "😢", "Stressed": "😰", "Excited": "🔥",
             "Romantic": "❤️", "Bored": "🥱"}
    return st.selectbox(label, utils.MOODS, index=None,
                        format_func=lambda m: f"{icons[m]} {m}",
                        placeholder="Choose your mood...", key=key)


def watched_expander(user_id):
    history = data_loader.get_user_history(user_id, RATINGS, LOOKUP)
    with st.expander(f"🎬 Already watched ({len(history)} movies) — these are excluded from recommendations"):
        if history.empty:
            st.write("No ratings recorded for this user yet.")
        else:
            st.dataframe(history, height=250, use_container_width=True)


# ── Page header ───────────────────────────────────────────────────
st.title("🎬 Mood Based Recommendation System")
st.caption("Select who you are and how you "
           "feel, and receive personalised movie recommendations.")

tab_rec, tab_compare, tab_eval = st.tabs(
    ["🎬 Movie Recommendations", "⚖️ Algorithm Comparison", "📊 Evaluation Results"])

# ── Tab 1: main recommendation page ───────────────────────────────
with tab_rec:
    user_id = user_selector("rec_user")
    watched_expander(user_id)
    mood = mood_selector("rec_mood")
    n = st.select_slider("Number of recommendations", utils.N_OPTIONS, value=utils.DEFAULT_N,
                         key="rec_n")
    method = st.selectbox("Algorithm", list(ALGORITHM_KEYS), index=2, key="rec_method")

    if st.button("🎬 Recommend Movies", type="primary", use_container_width=True, key="rec_btn"):
        if mood is None:
            st.warning("Please select your current mood.")
        else:
            with st.spinner("Finding movies for you..."):
                result, meta = generate_recommendations(user_id, mood, n, method)
                st.session_state["rec_result"] = (result, meta, user_id, mood, n, method)

    if "rec_result" in st.session_state:
        result, meta, r_user, r_mood, r_n, r_method = st.session_state["rec_result"]
        st.divider()
        st.subheader(f"Recommended for user {r_user} — feeling {r_mood}")
        if meta.get("notice"):
            st.info(meta["notice"])
        if meta.get("empty") or result.empty:
            st.warning("No suitable movies were found for this selection. Please try another mood.")
        else:
            render_result_cards(result, r_mood, r_method)

# ── Tab 2: algorithm comparison (plan §28) ────────────────────────
with tab_compare:
    st.caption("The same user and mood run through each group member's algorithm.")
    c_user = user_selector("cmp_user")
    c_mood = mood_selector("cmp_mood")
    c_n = st.select_slider("Number of recommendations", utils.N_OPTIONS,
                           value=utils.DEFAULT_N, key="cmp_n")

    if st.button("⚖️ Compare All Algorithms", type="primary", use_container_width=True, key="cmp_btn"):
        if c_mood is None:
            st.warning("Please select your current mood.")
        else:
            with st.spinner("Running all algorithms..."):
                st.session_state["cmp_result"] = (
                    {m: generate_recommendations(c_user, c_mood, c_n, m)[0]
                     for m in ALGORITHM_KEYS}, c_user, c_mood, c_n)

    if "cmp_result" in st.session_state:
        results, r_user, r_mood, r_n = st.session_state["cmp_result"]
        st.divider()
        st.subheader(f"User {r_user} — feeling {r_mood} — Top-{r_n}")
        columns = st.columns(3)
        for col, (method, result) in zip(columns, results.items()):
            with col:
                st.markdown(f"**{method}**")
                if result.empty:
                    st.warning("No results.")
                    continue
                for _, movie in result.iterrows():
                    genres = str(movie["genres_str"]).replace("|", " • ")
                    score = "pop." if movie["source"] == "popularity" else f"{movie['score']:.2f}"
                    st.markdown(f"**#{movie['rank']} {movie['title']}**")
                    st.caption(f"{genres}  \n⭐ {score} / 5")

# ── Tab 3: evaluation results (plan §30, §31) ─────────────────────
with tab_eval:

    metrics_path = utils.OUTPUT_DIR / "metrics_rating.csv"
    topk_path = utils.OUTPUT_DIR / "metrics_topk.csv"

    st.subheader("Rating Prediction Accuracy")
    if metrics_path.exists():
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
    else:
        st.info("Run scripts/run_evaluation.py to generate the metrics.")

    for chart, caption in [
        ("evaluation_results.png", "Rating-prediction error per algorithm: the hybrid "
                                   "combines both models and achieves the lowest RMSE/MAE."),
        ("error_distribution.png", "Distribution of prediction errors (actual − predicted) per "
                                   "algorithm; a peak at zero means accurate predictions."),
        ("alpha_tuning.png", "Tuning the hybrid blending weight α: the chosen α balances "
                             "collaborative and content-based contributions."),
    ]:
        path = utils.OUTPUT_DIR / chart
        if path.exists():
            st.image(path, caption=caption, use_container_width=True)

    st.subheader("Top-N Recommendation Quality")
    if topk_path.exists():
        st.dataframe(pd.read_csv(topk_path), use_container_width=True)
    else:
        st.info("Run scripts/run_evaluation.py to generate the metrics.")

    for chart, caption in [
        ("precision_recall_f1.png", "Precision@K / Recall@K / F1@K of the hybrid: of the "
                                    "movies recommended, how many the user actually liked."),
        ("mood_recommendations.png", "Top-rated movies per mood category, showing the "
                                     "mood-to-genre mapping produces distinct candidate pools."),
    ]:
        path = utils.OUTPUT_DIR / chart
        if path.exists():
            st.image(path, caption=caption, use_container_width=True)
