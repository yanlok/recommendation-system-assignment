"""Collaborative filtering - adapted from Member A's
"collaborative recommend.ipynb".

Methodology preserved from the notebook:
- user-user cosine similarity on mean-centred ratings (pivot -> subtract row
  mean -> fillna(0) -> cosine);
- prediction = weighted average of the RAW ratings of positively-similar
  users; the recommendation path uses the top-50 most similar users
  (notebook cell 23), the evaluation path uses all positive-similarity users
  (notebook cell 40);
- no prediction (NaN) when no similar user rated the movie.

Adaptation for the prototype: the notebook's dense pivot table is replaced by
the precomputed CSR matrix + similarity artifacts in models/ (identical
maths, feasible memory footprint).
"""

import numpy as np
import pandas as pd

from src import utils


class CollaborativeFilter:
    def __init__(self, user_item, user_sim, user_ids, movie_ids):
        self.user_item = user_item.tocsr()
        self.user_sim = np.asarray(user_sim)
        self.user_ids = list(user_ids)
        self.movie_ids = list(movie_ids)
        self.user_pos = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_pos = {mid: i for i, mid in enumerate(self.movie_ids)}

    def _target_row(self, user_id):
        uidx = self.user_pos.get(user_id)
        if uidx is None:
            raise KeyError(f"Unknown user: {user_id}")
        return uidx

    def _top_neighbours(self, uidx, k=utils.CF_TOP_K):
        """Indices and similarities of the k most similar positive users."""
        sims = self.user_sim[uidx].astype(np.float32)
        sims[uidx] = 0.0  # never compare the user with themselves
        sims[sims <= 0] = 0.0
        k = min(k, int((sims > 0).sum()))
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        top = np.argpartition(sims, -k)[-k:]
        return top, sims[top]

    def score_candidates(self, user_id, candidate_ids):
        """Predicted rating for each candidate (top-50 neighbour variant).

        Returns a pd.Series indexed by ratingId; NaN where no similar user
        rated the movie (faithful to the notebook, filled downstream).
        """
        uidx = self._target_row(user_id)
        cand_cols = np.array(
            [self.movie_pos[m] for m in candidate_ids if m in self.movie_pos], dtype=np.int64
        )
        if cand_cols.size == 0:
            return pd.Series(dtype=np.float64)

        nbr_idx, nbr_sims = self._top_neighbours(uidx)
        if nbr_idx.size == 0:
            return pd.Series(np.nan, index=candidate_ids)

        block = self.user_item[nbr_idx][:, cand_cols].toarray().astype(np.float32)
        rated = block > 0
        weights = np.where(rated, nbr_sims[:, None], 0.0)
        num = (block * weights).sum(axis=0)
        den = weights.sum(axis=0)

        preds = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        matched = [m for m in candidate_ids if m in self.movie_pos]
        return pd.Series(preds, index=matched)

    def predict_pairs(self, user_id, movie_ids):
        """Notebook cell-40 variant: all positive-similarity users, no top-k cap.

        Used by the RMSE/MAE evaluation; NaN when no similar user rated.
        """
        uidx = self._target_row(user_id)
        sims = self.user_sim[uidx].astype(np.float32)
        sims[uidx] = 0.0

        matched = [m for m in movie_ids if m in self.movie_pos]
        if not matched:
            return {m: np.nan for m in movie_ids}
        cols = np.array([self.movie_pos[m] for m in matched], dtype=np.int64)

        block = self.user_item[:, cols].toarray().astype(np.float32)  # users x movies
        positive_sim = (sims > 0)[:, None]
        weights = np.where((block > 0) & positive_sim, sims[:, None], 0.0)
        num = (block * weights).sum(axis=0)
        den = weights.sum(axis=0)
        preds = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)

        result = {m: np.nan for m in movie_ids}
        result.update(dict(zip(matched, (float(p) for p in preds))))
        return result
