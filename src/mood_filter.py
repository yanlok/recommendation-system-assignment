"""Mood-to-genre filtering (plan §9, §18 steps 2-4).

The mood -> genre mapping lives in data/processed/mood_genre_mapping.csv and
is the single source of truth - never hard-code the mapping elsewhere.
"""

import pandas as pd

from src import utils


def load_mood_genres() -> dict[str, list[str]]:
    """Return {mood: [genre, ...]} from the mapping CSV."""
    mapping = pd.read_csv(utils.DATA_DIR / "mood_genre_mapping.csv")
    return {row["mood"]: row["genres"].split("|") for _, row in mapping.iterrows()}


def mood_candidate_ids(mood: str, movie_lookup: pd.DataFrame) -> list[int]:
    """RatingIds of every movie matching at least one genre of the mood."""
    genres = load_mood_genres()[mood]
    genre_sets = movie_lookup["genres_str"].fillna("").str.split("|").apply(set)
    mask = genre_sets.apply(lambda movie_genres: any(g in movie_genres for g in genres))
    return movie_lookup.loc[mask, "ratingId"].tolist()
