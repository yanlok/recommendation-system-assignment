"""Loading of precomputed artifacts and display data for the app.

Plain functions (no Streamlit imports) so scripts can reuse them; app.py
wraps them in st.cache_resource / st.cache_data.
"""

import json

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src import utils


def load_movie_lookup() -> pd.DataFrame:
    """Light movie table for display and mood filtering (never movies_clean.csv)."""
    return pd.read_csv(utils.DATA_DIR / "movie_lookup.csv").drop_duplicates("ratingId")


def load_ratings_small() -> pd.DataFrame:
    """Rating history (userId, ratingId, rating) used for the watched list."""
    return pd.read_csv(utils.DATA_DIR / "ratings_clean.csv",
                       usecols=["userId", "ratingId", "rating"])


def load_artifacts() -> dict:
    """Materialised precomputed matrices from models/ (never lazy NpzFiles)."""
    with open(utils.MODEL_DIR / "id_maps.json", encoding="utf-8") as f:
        id_maps = json.load(f)

    def dense(name):
        npz = np.load(utils.MODEL_DIR / f"{name}.npz")
        array = npz["sim"]
        npz.close()
        return array

    return {
        "user_item": load_npz(utils.MODEL_DIR / "user_item.npz").tocsr(),
        "user_means": np.load(utils.MODEL_DIR / "user_means.npy"),
        "user_sim_cf": dense("user_sim_cf"),
        "user_sim_raw": dense("user_sim_raw"),
        "item_sim_content": dense("item_sim_content"),
        "item_sim_hybrid": dense("item_sim_hybrid"),
        "user_ids": id_maps["user_ids"],
        "movie_ids": id_maps["movie_ids"],
        "global_mean": id_maps["global_mean"],
    }


def get_user_history(user_id, ratings_small: pd.DataFrame, movie_lookup: pd.DataFrame) -> pd.DataFrame:
    """Movies the user already rated, best-rated first (for the watched list)."""
    history = ratings_small.loc[ratings_small["userId"] == user_id, ["ratingId", "rating"]]
    history = history.merge(
        movie_lookup[["ratingId", "title", "genres_str"]], on="ratingId", how="left"
    )
    return history[["title", "genres_str", "rating"]].sort_values("rating", ascending=False)
