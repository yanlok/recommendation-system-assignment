"""Builders for the user-item matrix and similarity matrices.

Used offline by scripts/prepare_models.py (full-data artifacts for the app)
and scripts/run_evaluation.py (train-split matrices for evaluation). The
Streamlit app only loads the precomputed artifacts - it never rebuilds these.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item(ratings_df, user_ids, movie_ids):
    """CSR user-item matrix of raw ratings over the given id spaces.

    Duplicate (userId, ratingId) rows are averaged, matching the pivot-table
    behaviour of the original notebooks (csr_matrix would sum them).
    """
    deduped = ratings_df.groupby(["userId", "ratingId"], as_index=False)["rating"].mean()
    user_map = {uid: i for i, uid in enumerate(user_ids)}
    movie_map = {mid: i for i, mid in enumerate(movie_ids)}

    mask = deduped["userId"].isin(user_map) & deduped["ratingId"].isin(movie_map)
    rows = deduped.loc[mask, "userId"].map(user_map).to_numpy()
    cols = deduped.loc[mask, "ratingId"].map(movie_map).to_numpy()
    values = deduped.loc[mask, "rating"].to_numpy(dtype=np.float32)

    matrix = csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(movie_ids)))
    matrix.indices = matrix.indices.astype(np.int32)
    matrix.indptr = matrix.indptr.astype(np.int32)
    return matrix


def compute_user_means(user_item):
    """Mean rating per user (users with no ratings default to 3.5)."""
    sums = np.asarray(user_item.sum(axis=1)).flatten()
    counts = np.asarray((user_item > 0).sum(axis=1)).flatten()
    means = np.divide(sums, counts, out=np.full(len(sums), 3.5, dtype=np.float64), where=counts > 0)
    return means.astype(np.float32)


def compute_user_sim_raw(user_item):
    """Cosine similarity between users on raw ratings (hybrid's CF, run_hybrid.py)."""
    return cosine_similarity(user_item).astype(np.float32)


def compute_user_sim_centered(user_item, user_means):
    """Cosine similarity between users on mean-centred ratings (friend A's CF).

    Equivalent to the notebook's pivot -> subtract row mean -> fillna(0) ->
    cosine similarity: unrated entries stay 0 (neutral) after centring.
    """
    centered = user_item.copy()
    centered.data = centered.data - user_means[centered.nonzero()[0]]
    return cosine_similarity(centered).astype(np.float32)
