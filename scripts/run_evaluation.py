"""Offline evaluation of the three algorithms (plan §29-§32).

Run from the repo root:  python scripts/run_evaluation.py

Rebuilds the train/test split (80/20, random_state=42 - the shared settings
every member used) and evaluates the ADAPTED implementations in src/, so the
charts in output/ and the numbers on the app's evaluation tab match what the
app actually computes. Writes:

  output/metrics_rating.csv      RMSE / MAE per method (+ valid prediction counts)
  output/metrics_topk.csv        Precision@K / Recall@K / F1@K of the hybrid
  output/evaluation_results.png, error_distribution.png, alpha_tuning.png,
  output/precision_recall_f1.png, mood_recommendations.png
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import collaborative, content_based, data_loader, hybrid, matrices, mood_filter, utils  # noqa: E402

K_VALUES = [5, 10, 20]
EVAL_USERS = 100


def collect_predictions(model, evaluation_rows):
    """{row_label: prediction} for each (user, movie) row, batched per user."""
    predictions = pd.Series(np.nan, index=evaluation_rows.index)
    for user_id, rows in evaluation_rows.groupby("userId"):
        pairs = model.predict_pairs(user_id, rows["ratingId"].tolist())
        predictions.loc[rows.index] = [pairs.get(mid, np.nan) for mid in rows["ratingId"]]
    return predictions


def rmse_mae(actual, predicted):
    valid = predicted.notna()
    pred = predicted[valid]
    act = actual[valid]
    rmse = float(np.sqrt(np.mean((act - pred) ** 2)))
    mae = float(np.mean(np.abs(act - pred)))
    return rmse, mae, int(valid.sum())


def chart_evaluation_results(results, actual, hybrid_pred, alpha):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    methods = list(results)
    x = np.arange(len(methods))
    axes[0].bar(x - 0.2, [results[m]["rmse"] for m in methods], 0.4, label="RMSE", color="steelblue")
    axes[0].bar(x + 0.2, [results[m]["mae"] for m in methods], 0.4, label="MAE", color="coral")
    axes[0].set_xticks(x, methods)
    axes[0].set_ylabel("Error"); axes[0].set_title("Rating Prediction Error")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    values = [results[m]["rmse"] for m in methods]
    axes[1].bar(["CF Only", "CBF Only", f"Hybrid (a={alpha:.1f})"], values,
                color=["#ff9999", "#66b3ff", "#99ff99"], edgecolor="black")
    axes[1].set_ylabel("RMSE"); axes[1].set_title("RMSE: Hybrid vs Individual")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)

    axes[2].scatter(actual, hybrid_pred, alpha=0.1, s=5)
    axes[2].plot([0.5, 5], [0.5, 5], "r--", linewidth=2)
    axes[2].set_xlabel("Actual Rating"); axes[2].set_ylabel("Predicted Rating")
    axes[2].set_title("Hybrid: Predicted vs Actual")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(utils.OUTPUT_DIR / "evaluation_results.png", dpi=150)
    plt.close(fig)


def chart_error_distribution(results, actual, predictions):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, color) in zip(axes, zip(results, ["steelblue", "coral", "green"])):
        valid = predictions[name].notna()
        errors = actual[valid] - predictions[name][valid]
        ax.hist(errors, bins=50, color=color, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_xlabel("Error"); ax.set_ylabel("Frequency")
        ax.set_title(f"{name} (mean={errors.mean():.3f}, std={errors.std():.3f})")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(utils.OUTPUT_DIR / "error_distribution.png", dpi=150)
    plt.close(fig)


def chart_alpha_tuning(alphas, rmses, maes, best_alpha):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas, rmses, "bo-", label="RMSE", linewidth=2)
    ax.plot(alphas, maes, "rs-", label="MAE", linewidth=2)
    ax.axvline(x=best_alpha, color="green", linestyle="--", label=f"Best: {best_alpha:.1f}")
    ax.set_xlabel("Alpha (CF weight)"); ax.set_ylabel("Error")
    ax.set_title("Hybrid Blending Weight Optimisation")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(utils.OUTPUT_DIR / "alpha_tuning.png", dpi=150)
    plt.close(fig)


def chart_precision_recall_f1(metrics_by_k):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ks = sorted(metrics_by_k)
    for ax, metric, title in zip(axes, ["precision", "recall", "f1"],
                                 ["Precision@K", "Recall@K", "F1@K"]):
        values = [metrics_by_k[k][metric] for k in ks]
        ax.plot(ks, values, "o-", color="steelblue", linewidth=2, markersize=8)
        for k, v in zip(ks, values):
            ax.text(k, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
        ax.set_xlabel("K"); ax.set_ylabel(title); ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(utils.OUTPUT_DIR / "precision_recall_f1.png", dpi=150)
    plt.close(fig)


def chart_mood_recommendations(movie_lookup):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    genre_sets = movie_lookup["genres_str"].fillna("").str.split("|").apply(set)
    for idx, (mood, genres) in enumerate(mood_filter.load_mood_genres().items()):
        ax = axes[idx // 3][idx % 3]
        mask = genre_sets.apply(lambda movie_genres: any(g in movie_genres for g in genres))
        mood_movies = movie_lookup[mask].nlargest(15, "avg_rating")
        ax.barh(range(len(mood_movies)), mood_movies["avg_rating"], color=plt.cm.Set2(idx))
        ax.set_yticks(range(len(mood_movies)), mood_movies["title"], fontsize=7)
        ax.set_xlabel("Avg Rating")
        ax.set_title(f"{mood} ({mask.sum()} movies)")
        ax.set_xlim(3.0, 4.5); ax.invert_yaxis()
    fig.suptitle("Top-Rated Movies by Mood Category", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(utils.OUTPUT_DIR / "mood_recommendations.png", dpi=150)
    plt.close(fig)


def main():
    start = time.time()
    utils.OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading data ...")
    ratings = pd.read_csv(utils.DATA_DIR / "ratings_clean.csv",
                          usecols=["userId", "ratingId", "rating"])
    movie_lookup = data_loader.load_movie_lookup()
    artifacts = data_loader.load_artifacts()  # item similarities are split-independent

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    evaluation_rows = (test.groupby("userId")
                       .filter(lambda rows: len(rows) >= 3)
                       .sample(n=8000, random_state=42))
    print(f"  train {len(train):,} / test {len(test):,} / RMSE-MAE sample {len(evaluation_rows):,}")

    user_ids = sorted(train["userId"].unique())
    movie_ids = sorted(train["ratingId"].unique())
    user_item = matrices.build_user_item(train, user_ids, movie_ids)
    user_means = matrices.compute_user_means(user_item)
    models = {
        "CF": collaborative.CollaborativeFilter(
            user_item, matrices.compute_user_sim_centered(user_item, user_means),
            user_ids, movie_ids),
        "CBF": content_based.ContentBasedFilter(
            user_item, artifacts["item_sim_content"], user_ids, movie_ids),
        "Hybrid": hybrid.HybridRecommender(
            user_item, matrices.compute_user_sim_raw(user_item),
            artifacts["item_sim_hybrid"], user_ids, movie_ids,
            global_mean=float(train["rating"].mean())),
    }

    print("Predicting RMSE/MAE sample ...")
    actual = evaluation_rows["rating"]
    predictions = {name: collect_predictions(model, evaluation_rows)
                   for name, model in models.items()}
    results = {}
    for name, predicted in predictions.items():
        results[name] = dict(zip(["rmse", "mae", "n_valid"], rmse_mae(actual, predicted)))
        print(f"  {name:<7} RMSE={results[name]['rmse']:.4f} MAE={results[name]['mae']:.4f} "
              f"(n={results[name]['n_valid']:,})")

    pd.DataFrame([
        {"method": name, "rmse": round(r["rmse"], 4), "mae": round(r["mae"], 4),
         "n_valid": r["n_valid"]}
        for name, r in results.items()
    ]).to_csv(utils.OUTPUT_DIR / "metrics_rating.csv", index=False)

    print("Alpha tuning ...")
    alphas = np.round(np.arange(0.0, 1.05, 0.1), 1)
    # Tune on the hybrid's own CF/CBF outputs (global-mean fallback -> no NaN)
    hybrid_model = models["Hybrid"]
    cf_parts, cbf_parts = [], []
    for user_id, rows in evaluation_rows.groupby("userId"):
        movie_ids_user = rows["ratingId"].tolist()
        cf_s = hybrid_model.cf_scores(user_id, movie_ids_user)
        cbf_s = hybrid_model.cbf_scores(user_id, movie_ids_user)
        cf_parts.append(pd.Series([cf_s.get(m, np.nan) for m in movie_ids_user], index=rows.index))
        cbf_parts.append(pd.Series([cbf_s.get(m, np.nan) for m in movie_ids_user], index=rows.index))
    cf_series = pd.concat(cf_parts)
    cbf_series = pd.concat(cbf_parts)

    rmses, maes = [], []
    for a in alphas:
        blended = a * cf_series + (1 - a) * cbf_series
        r, m, _ = rmse_mae(actual, blended)
        rmses.append(r); maes.append(m)
        print(f"  alpha={a:.1f}: RMSE={r:.4f}, MAE={m:.4f}")
    best_alpha = float(alphas[int(np.argmin(rmses))])
    print(f"  optimal alpha: {best_alpha:.1f} (app uses {utils.HYBRID_ALPHA})")

    chart_alpha_tuning(alphas, rmses, maes, best_alpha)
    chart_evaluation_results(results, actual, predictions["Hybrid"], utils.HYBRID_ALPHA)
    chart_error_distribution(results, actual, predictions)
    print("  charts saved")

    print(f"Top-K metrics for {EVAL_USERS} users ...")
    user_test = test.groupby("userId").agg(
        test_movies=("ratingId", list), test_ratings=("rating", list)).reset_index()
    eval_users = user_test[user_test["test_movies"].apply(len) >= 5].head(EVAL_USERS)

    metrics_by_k = {k: {"precision": [], "recall": [], "f1": []} for k in K_VALUES}
    for _, row in eval_users.iterrows():
        uid = row["userId"]
        relevant = {m for m, r in zip(row["test_movies"], row["test_ratings"])
                    if r >= utils.RELEVANCE_THRESHOLD}
        if not relevant:
            continue
        rated = set(train.loc[train["userId"] == uid, "ratingId"])
        candidates = [m for m in movie_ids if m not in rated]
        scores = models["Hybrid"].score_candidates(uid, candidates)
        ranked = scores.sort_values(ascending=False).index
        for k in K_VALUES:
            hits = len(set(ranked[:k]) & relevant)
            precision = hits / k
            recall = hits / len(relevant)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics_by_k[k]["precision"].append(precision)
            metrics_by_k[k]["recall"].append(recall)
            metrics_by_k[k]["f1"].append(f1)

    topk_rows = []
    for k in K_VALUES:
        averages = {m: float(np.mean(v)) if v else 0.0 for m, v in metrics_by_k[k].items()}
        topk_rows.append({"k": k, "precision": round(averages["precision"], 4),
                          "recall": round(averages["recall"], 4), "f1": round(averages["f1"], 4),
                          "users_evaluated": len(metrics_by_k[k]["precision"])})
        print(f"  K={k}: P={averages['precision']:.4f} R={averages['recall']:.4f} "
              f"F1={averages['f1']:.4f}")
    pd.DataFrame(topk_rows).to_csv(utils.OUTPUT_DIR / "metrics_topk.csv", index=False)
    chart_precision_recall_f1({k: {m: float(np.mean(v)) if v else 0.0
                                   for m, v in metrics_by_k[k].items()} for k in K_VALUES})
    chart_mood_recommendations(movie_lookup)

    print(f"Done in {time.time() - start:.1f}s - charts and metrics in {utils.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
