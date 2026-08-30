"""
Select demo users for the Streamlit prototype.

Picks three users with very different rating histories so the lecturer demo
can show strong personalisation (heavy), typical behaviour (medium) and the
cold-start fallback (sparse). Run once, then hard-code the result in src/utils.py.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ratings = pd.read_csv(ROOT / "data/processed/ratings_clean.csv",
                      usecols=["userId", "ratingId", "rating"])

counts = ratings.groupby("userId").agg(
    n_ratings=("rating", "size"),
    mean_rating=("rating", "mean"),
)
print(f"Users: {len(counts)}")
print(counts["n_ratings"].describe().round(1).to_string(), "\n")

def show(label, user_ids):
    print(f"--- {label} candidates ---")
    sample = counts.loc[user_ids].sort_values("n_ratings", ascending=False)
    print(sample.round(2).to_string(), "\n")

# Heavy: near the top but not the single extreme outlier
heavy_zone = counts[counts["n_ratings"] >= counts["n_ratings"].quantile(0.98)]
show("Heavy (>= p98)", heavy_zone.head(8).index.tolist())

# Medium: around the median
median = counts["n_ratings"].median()
medium_zone = counts[(counts["n_ratings"] - median).abs() <= 2]
show("Medium (~median)", medium_zone.head(8).index.tolist())

# Sparse: few ratings but enough for CF to still produce something
sparse_zone = counts[(counts["n_ratings"] >= 10) & (counts["n_ratings"] <= 20)]
show("Sparse (10-20 ratings)", sparse_zone.head(8).index.tolist())
