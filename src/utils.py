"""Shared constants and helper functions for the recommendation system."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"

# ── Configuration (single source of truth, see plan §36) ─────────
HYBRID_ALPHA = 0.5            # weight of collaborative score in the hybrid blend
CF_TOP_K = 50                 # neighbours / raters used by collaborative predictors
CBF_TOP_K = 30                # rated movies used by the hybrid's content predictor
RELEVANCE_THRESHOLD = 3.5     # rating >= threshold counts as relevant
DEFAULT_N = 10                # default number of recommendations
N_OPTIONS = [5, 10, 15, 20]
SPARSE_USER_THRESHOLD = 20    # users with fewer ratings get the popularity fallback
POPULARITY_MIN_VOTES = 100    # Bayesian prior strength for the popularity score

MOODS = ["Happy", "Sad", "Stressed", "Excited", "Romantic", "Bored"]

# Pre-tested demo users (see scripts/pick_demo_users.py)
DEMO_USERS = [
    ("Demo User A", 19886, "Heavy rater (1,546 ratings) - strong personalisation"),
    ("Demo User B", 26812, "Medium rater (198 ratings) - typical user"),
    ("Demo User C", 2227, "Sparse rater (18 ratings) - cold-start fallback demo"),
]


def popularity_score(movie_lookup: pd.DataFrame) -> pd.Series:
    """Bayesian-weighted average rating, indexed by ratingId.

    Used as the fallback ranking for cold-start/sparse users and to fill
    candidate shortfalls (plan §19.1): a 5.0 with 1 vote must not outrank
    a 4.3 with 1,000 votes.
    """
    v = movie_lookup["num_ratings"].fillna(0).astype(float)
    m = POPULARITY_MIN_VOTES
    C = float(movie_lookup["avg_rating"].mean())
    R = movie_lookup["avg_rating"].fillna(C)
    weighted = (v / (v + m)) * R + (m / (v + m)) * C
    return pd.Series(weighted.values, index=movie_lookup["ratingId"].values)


def mood_match_note(mood: str, genres: list[str]) -> str:
    return f"Matches your {mood} mood ({', '.join(genres)})"


def reason_for(method: str) -> str:
    """Short per-algorithm explanation shown on each result card (plan §13)."""
    return {
        "collaborative": "Users with a similar rating history enjoyed this movie.",
        "content": "Similar in genres, keywords and cast to movies you rated highly.",
        "hybrid": f"Blends similar-user ratings ({HYBRID_ALPHA:.0%}) and content similarity ({1 - HYBRID_ALPHA:.0%}).",
        "popularity": "Based on your mood and overall movie popularity.",
    }[method]
