"""Hybrid recommendation - adapted from the owner's src/run_hybrid.py.

Methodology preserved from run_hybrid.py:
- CF: weighted average of the raw ratings of the top-50 users most similar
  (raw-ratings cosine) among those who rated the movie; global-mean fallback
  when there is no signal;
- CBF: weighted average of the user's ratings over the top-30 movies most
  similar (500-feature TF-IDF cosine) among those the user rated, positive
  similarities only; global-mean fallback;
- hybrid score = alpha * CF + (1 - alpha) * CBF with alpha = 0.5.

All three predictors are vectorised here but the maths is identical to the
original batch_cf_predict / batch_cbf_predict.
"""

import numpy as np
import pandas as pd

from src import utils


class HybridRecommender:
    def __init__(self, user_item, user_sim, item_sim, user_ids, movie_ids, global_mean):
        self.user_item = user_item.tocsr()
        self.user_sim = np.asarray(user_sim)
        self.item_sim = np.asarray(item_sim)
        self.user_ids = list(user_ids)
        self.movie_ids = list(movie_ids)
        self.user_pos = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_pos = {mid: i for i, mid in enumerate(self.movie_ids)}
        self.global_mean = float(global_mean)

    def _target_idx(self, user_id):
        uidx = self.user_pos.get(user_id)
        if uidx is None:
            raise KeyError(f"Unknown user: {user_id}")
        return uidx

    def _resolve_candidates(self, candidate_ids):
        cand_idx = np.array(
            [self.movie_pos[m] for m in candidate_ids if m in self.movie_pos], dtype=np.int64
        )
        matched = [m for m in candidate_ids if m in self.movie_pos]
        return cand_idx, matched

    def cf_scores(self, user_id, candidate_ids):
        """Vectorised batch_cf_predict: top-50 similar raters per movie."""
        uidx = self._target_idx(user_id)
        cand_idx, matched = self._resolve_candidates(candidate_ids)
        if cand_idx.size == 0:
            return pd.Series(dtype=np.float64)

        raters = self.user_item[:, cand_idx].toarray().astype(np.float32)  # users x candidates
        sims = self.user_sim[uidx].astype(np.float32)

        # -inf marks "did not rate" so it can never enter the top-k raters
        weighted = np.where(raters > 0, sims[:, None], -np.inf)
        k = min(utils.CF_TOP_K, raters.shape[0])
        top = np.argpartition(weighted, -k, axis=0)[-k:, :]
        sims_top = np.take_along_axis(weighted, top, axis=0)
        ratings_top = np.take_along_axis(raters, top, axis=0)

        sims_top = np.where(np.isfinite(sims_top), sims_top, 0.0)
        num = (sims_top * ratings_top).sum(axis=0)
        den = np.abs(sims_top).sum(axis=0) + 1e-8
        sim_sum = sims_top.sum(axis=0)  # run_hybrid's condition: signed sum > 0
        preds = np.where(sim_sum > 0, num / den, self.global_mean)
        return pd.Series(preds, index=matched)

    def cbf_scores(self, user_id, candidate_ids):
        """Vectorised batch_cbf_predict: top-30 similar rated movies per candidate."""
        self._target_idx(user_id)
        rated_cols, ratings = self._user_ratings(user_id)
        cand_idx, matched = self._resolve_candidates(candidate_ids)
        if cand_idx.size == 0:
            return pd.Series(dtype=np.float64)
        if rated_cols.size == 0:
            return pd.Series(self.global_mean, index=matched)

        sims = self.item_sim[np.array(cand_idx)][:, rated_cols].astype(np.float32)  # cand x rated
        k = min(utils.CBF_TOP_K, rated_cols.size)
        top = np.argpartition(sims, -k, axis=1)[:, -k:]
        sims_top = np.take_along_axis(sims, top, axis=1)
        ratings_top = ratings[top]

        positive = sims_top > 0
        num = np.where(positive, sims_top * ratings_top, 0.0).sum(axis=1)
        den = np.abs(np.where(positive, sims_top, 0.0)).sum(axis=1) + 1e-8
        preds = np.where(positive.any(axis=1), num / den, self.global_mean)
        return pd.Series(preds, index=matched)

    def _user_ratings(self, user_id):
        uidx = self._target_idx(user_id)
        row = self.user_item[uidx].tocoo()
        order = np.argsort(row.data)[::-1]
        return row.col[order], row.data[order].astype(np.float32)

    def score_candidates(self, user_id, candidate_ids):
        """alpha * CF + (1 - alpha) * CBF, indexed by ratingId."""
        cf = self.cf_scores(user_id, candidate_ids)
        cbf = self.cbf_scores(user_id, candidate_ids)
        return utils.HYBRID_ALPHA * cf + (1 - utils.HYBRID_ALPHA) * cbf

    def predict_pairs(self, user_id, movie_ids):
        """{movieId: prediction} for evaluation (fallbacks mean no NaN)."""
        return self.score_candidates(user_id, list(movie_ids)).to_dict()
