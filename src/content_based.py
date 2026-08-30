"""Content-based filtering - adapted from Member B's notebook/content.py.

Methodology preserved from the notebook:
- TF-IDF over genres + keywords + overview + cast + original language
  (max_features=20000, English stop words) and item-item cosine similarity;
- rating prediction (notebook §8) = similarity-weighted average of the user's
  ratings over ALL positively-similar rated movies, clipped to [0.5, 5.0];
- NaN when the user has no history or no positive similarity exists.

Adaptations for the prototype: the notebook works in tmdbId space and
recomputes TF-IDF at runtime - here everything runs in ratingId space against
the precomputed item-similarity artifact, and candidates are ranked with the
same §8 prediction formula (the notebook's alternative liked-seed ranking,
§10, stays in the notebook). Feature building itself happens offline in
scripts/prepare_models.py.
"""

import numpy as np
import pandas as pd


class ContentBasedFilter:
    def __init__(self, user_item, item_sim, user_ids, movie_ids):
        self.user_item = user_item.tocsr()
        self.item_sim = np.asarray(item_sim)
        self.user_ids = list(user_ids)
        self.movie_ids = list(movie_ids)
        self.user_pos = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_pos = {mid: i for i, mid in enumerate(self.movie_ids)}

    def _user_ratings(self, user_id):
        """(rated movie column indices, raw ratings) for the user."""
        uidx = self.user_pos.get(user_id)
        if uidx is None:
            raise KeyError(f"Unknown user: {user_id}")
        row = self.user_item[uidx].tocoo()
        order = np.argsort(row.data)[::-1]
        return row.col[order], row.data[order].astype(np.float32)

    def score_candidates(self, user_id, candidate_ids):
        """Predicted rating for each candidate (notebook §8 formula).

        Returns a pd.Series indexed by ratingId; NaN where the formula has
        no positive-similarity coverage.
        """
        rated_cols, ratings = self._user_ratings(user_id)
        if rated_cols.size == 0:
            return pd.Series(np.nan, index=candidate_ids)

        cand_idx = [self.movie_pos[m] for m in candidate_ids if m in self.movie_pos]
        if not cand_idx:
            return pd.Series(dtype=np.float64)

        sims = self.item_sim[np.array(cand_idx)][:, rated_cols].astype(np.float32)
        positive = sims > 0
        num = np.where(positive, sims * ratings[None, :], 0.0).sum(axis=1)
        den = np.where(positive, sims, 0.0).sum(axis=1)

        preds = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        preds = np.clip(preds, 0.5, 5.0)
        matched = [m for m in candidate_ids if m in self.movie_pos]
        return pd.Series(preds, index=matched)

    def predict_pairs(self, user_id, movie_ids):
        """Same §8 formula, returned as {movieId: prediction} for evaluation."""
        return self.score_candidates(user_id, list(movie_ids)).to_dict()
