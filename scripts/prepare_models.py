"""Offline preparation of every artifact the app needs (plan §23).

Run once from the repo root:  python scripts/prepare_models.py

Outputs in models/:
  user_item.npz         CSR matrix of raw ratings (users x movies, float32)
  user_means.npy        mean rating per user
  user_sim_cf.npz       user-user cosine similarity on mean-centred ratings (Member A)
  user_sim_raw.npz      user-user cosine similarity on raw ratings (hybrid CF)
  item_sim_hybrid.npz   movie-movie cosine similarity, 500-feature TF-IDF (hybrid CBF)
  item_sim_content.npz  movie-movie cosine similarity, 20k-feature TF-IDF (Member B)
  id_maps.json          user/movie id ordering + global mean rating

All matrices are stored version-agnostic (.npz/.npy/.json, no pickles) so the
Streamlit Cloud runtime does not need scikit-learn or a matching version.
"""

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import matrices, utils  # noqa: E402


def parse_list(value):
    """Safely evaluate the stringified list columns."""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def build_item_similarity(movies, columns, max_features, name):
    text = movies[columns[0]]
    for col in columns[1:]:
        text = text + " " + movies[col]
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    features = tfidf.fit_transform(text)
    sim = cosine_similarity(features).astype(np.float16)
    path = utils.MODEL_DIR / f"{name}.npz"
    np.savez_compressed(path, sim=sim)
    print(f"  {name}: features {features.shape}, sim {sim.shape}, "
          f"{path.stat().st_size / 1e6:.1f} MB")
    return sim


def main():
    start = time.time()
    utils.MODEL_DIR.mkdir(exist_ok=True)

    print("Loading data ...")
    movie_lookup = pd.read_csv(utils.DATA_DIR / "movie_lookup.csv")
    # 4 movies appear twice in the processed files; keep one row per ratingId
    movie_lookup = movie_lookup.drop_duplicates("ratingId", keep="first")
    movie_ids = sorted(movie_lookup["ratingId"].unique())

    # Align the feature columns to the canonical movie ordering
    movies = pd.read_csv(
        utils.DATA_DIR / "movies_clean.csv",
        usecols=["ratingId", "genres_str", "genre_list", "keyword_list",
                 "overview", "cast", "original_language"],
    ).drop_duplicates("ratingId", keep="first").set_index("ratingId").reindex(movie_ids)
    missing = movies.index[movies["genres_str"].isna()].shape[0]
    if missing:
        raise SystemExit(f"{missing} movie_lookup ids missing from movies_clean.csv")
    movies["genres_str"] = movies["genres_str"].fillna("")
    movies["overview"] = movies["overview"].fillna("")
    movies["cast"] = movies["cast"].fillna("")
    movies["original_language"] = movies["original_language"].fillna("")

    ratings = pd.read_csv(utils.DATA_DIR / "ratings_clean.csv",
                          usecols=["userId", "ratingId", "rating"])
    user_ids = sorted(ratings["userId"].unique())
    global_mean = float(ratings["rating"].mean())
    print(f"  {len(user_ids)} users, {len(movie_ids)} movies, {len(ratings):,} ratings")

    print("Building item similarities (TF-IDF) ...")
    # Hybrid CBF: genres + top-20 keywords, 500 features (run_hybrid.py)
    genres = movies["genre_list"].apply(parse_list).apply(lambda g: " ".join(g))
    keywords = movies["keyword_list"].apply(parse_list).apply(lambda k: " ".join(k[:20]))
    movies["_hybrid_text"] = genres + " " + keywords
    build_item_similarity(movies, ["_hybrid_text"], 500, "item_sim_hybrid")

    # Member B CBF: genres + keywords + overview + cast + language, 20k features
    for col in ["keyword_list", "overview", "cast", "original_language"]:
        movies[col] = movies[col].fillna("").astype(str)
    build_item_similarity(movies, ["genres_str", "keyword_list", "overview", "cast",
                                   "original_language"], 20_000, "item_sim_content")

    print("Building user matrices ...")
    user_item = matrices.build_user_item(ratings, user_ids, movie_ids)
    user_means = matrices.compute_user_means(user_item)
    user_sim_raw = matrices.compute_user_sim_raw(user_item)
    user_sim_cf = matrices.compute_user_sim_centered(user_item, user_means)

    save_npz(utils.MODEL_DIR / "user_item.npz", user_item)
    np.save(utils.MODEL_DIR / "user_means.npy", user_means)
    for name, sim in [("user_sim_cf", user_sim_cf), ("user_sim_raw", user_sim_raw)]:
        np.savez_compressed(utils.MODEL_DIR / f"{name}.npz", sim=sim.astype(np.float16))
        print(f"  {name}: {sim.shape}, "
              f"{(utils.MODEL_DIR / f'{name}.npz').stat().st_size / 1e6:.1f} MB")

    with open(utils.MODEL_DIR / "id_maps.json", "w", encoding="utf-8") as f:
        json.dump({"user_ids": [int(u) for u in user_ids],
                   "movie_ids": [int(m) for m in movie_ids],
                   "global_mean": global_mean}, f)

    print(f"Done in {time.time() - start:.1f}s - artifacts in {utils.MODEL_DIR}")


if __name__ == "__main__":
    main()
