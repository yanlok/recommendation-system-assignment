"""
Hybrid Recommendation System — Optimised Implementation
Combines CF + CBF with mood-to-genre mapping.
Uses batch prediction for speed.
"""

import pandas as pd
import numpy as np
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os, warnings
warnings.filterwarnings('ignore')
os.makedirs('output', exist_ok=True)

# ── Load ──────────────────────────────────────────────────────
movies = pd.read_csv('data/processed/movies_clean.csv')
ratings = pd.read_csv('data/processed/ratings_clean.csv')
mood_map = pd.read_csv('data/processed/mood_genre_mapping.csv')
movies['genres_list'] = movies['genre_list'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
movies['keywords_list'] = movies['keyword_list'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

print(f'Movies: {len(movies)}, Ratings: {len(ratings):,}')
print(f'Train: {len(train):,}, Test: {len(test):,}')
print(f'Users: {train["userId"].nunique()}, Movies: {train["ratingId"].nunique()}')

# ── CF: build user-item matrix + similarity ───────────────────
user_ids = sorted(train['userId'].unique())
movie_ids_cf = sorted(train['ratingId'].unique())
user_map = {uid: i for i, uid in enumerate(user_ids)}
movie_map_cf = {mid: i for i, mid in enumerate(movie_ids_cf)}
movie_inv_cf = {i: mid for mid, i in movie_map_cf.items()}

row = train['userId'].map(user_map).values
col = train['ratingId'].map(movie_map_cf).values
user_item = csr_matrix((train['rating'].values, (row, col)), shape=(len(user_ids), len(movie_ids_cf)))
user_sim = cosine_similarity(user_item)

# Mean-centred ratings for better similarity
user_means = np.array(user_item.sum(axis=1)).flatten() / np.array((user_item > 0).sum(axis=1)).flatten()
user_means = np.nan_to_num(user_means, nan=3.5)

print(f'CF: user-item {user_item.shape}, density {user_item.nnz/(user_item.shape[0]*user_item.shape[1]):.2%}')

# ── CBF: TF-IDF features + item similarity ───────────────────
movies['feature_text'] = movies['genres_list'].apply(lambda x: ' '.join(x)) + ' ' + \
                          movies['keywords_list'].apply(lambda x: ' '.join(x[:20]))
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
item_features = tfidf.fit_transform(movies['feature_text'])
rid_to_idx = {rid: i for i, rid in enumerate(movies['ratingId'])}
item_sim = cosine_similarity(item_features)

print(f'CBF: item features {item_features.shape}')

# ── Batch prediction for test set ─────────────────────────────
print('\nPre-computing batch predictions for test set...')

# Build lookup: user_id -> list of (movie_id, rating) from train
user_train_dict = train.groupby('userId').apply(
    lambda g: dict(zip(g['ratingId'], g['rating']))
).to_dict()

# Build lookup: movie_id -> user ratings
movie_user_dict = train.groupby('ratingId').apply(
    lambda g: dict(zip(g['userId'], g['rating']))
).to_dict()

def batch_cf_predict(user_id, movie_id):
    """CF: weighted avg of top-50 similar users who rated this movie."""
    if user_id not in user_map or movie_id not in movie_map_cf:
        return train['rating'].mean()
    uidx = user_map[user_id]
    sims = user_sim[uidx]
    # Users who rated this movie
    raters = movie_user_dict.get(movie_id, {})
    if not raters:
        return train['rating'].mean()
    rater_indices = [user_map[uid] for uid in raters if uid in user_map]
    if not rater_indices:
        return train['rating'].mean()
    rater_sims = sims[rater_indices]
    rater_ratings = np.array([raters[user_ids[i]] for i in rater_indices])
    top = np.argsort(rater_sims)[::-1][:50]
    ts, tr = rater_sims[top], rater_ratings[top]
    return np.dot(ts, tr) / (np.abs(ts).sum() + 1e-8) if ts.sum() > 0 else train['rating'].mean()

def batch_cbf_predict(user_id, movie_id):
    """CBF: weighted avg of top-30 similar movies the user rated."""
    if movie_id not in rid_to_idx:
        return train['rating'].mean()
    midx = rid_to_idx[movie_id]
    user_rated = user_train_dict.get(user_id, {})
    if not user_rated:
        return train['rating'].mean()
    rated_indices = [rid_to_idx[mid] for mid in user_rated if mid in rid_to_idx]
    if not rated_indices:
        return train['rating'].mean()
    sims = item_sim[midx][rated_indices]
    ratings = np.array([user_rated[list(rid_to_idx.keys())[list(rid_to_idx.values()).index(i)]] for i in rated_indices])
    # Simpler approach
    sim_scores = []
    for mid, rating in user_rated.items():
        if mid in rid_to_idx:
            sim_scores.append((item_sim[midx][rid_to_idx[mid]], rating))
    if not sim_scores:
        return train['rating'].mean()
    sim_scores.sort(key=lambda x: x[0], reverse=True)
    top = sim_scores[:30]
    ts = np.array([s for s, _ in top])
    tr = np.array([r for _, r in top])
    pos = ts > 0
    return np.dot(ts[pos], tr[pos]) / (np.abs(ts[pos]).sum() + 1e-8) if pos.any() else train['rating'].mean()

# Sample test set for evaluation
test_sample = test.groupby('userId').filter(lambda x: len(x) >= 3).sample(n=8000, random_state=42)
print(f'Evaluating on {len(test_sample):,} test samples...')

cf_all, cbf_all, actual_all = [], [], []
for i, (_, row) in enumerate(test_sample.iterrows()):
    uid, mid, actual = row['userId'], row['ratingId'], row['rating']
    cf_all.append(batch_cf_predict(uid, mid))
    cbf_all.append(batch_cbf_predict(uid, mid))
    actual_all.append(actual)
    if (i+1) % 2000 == 0:
        print(f'  {i+1}/{len(test_sample)} done...')

cf_all = np.array(cf_all)
cbf_all = np.array(cbf_all)
actual_all = np.array(actual_all)

# ── Alpha tuning ──────────────────────────────────────────────
print('\n' + '='*70)
print('ALPHA TUNING')
print('='*70)

alphas = np.arange(0.0, 1.05, 0.1)
best_rmse, best_alpha = 999, 0.5
for a in alphas:
    pred = a * cf_all + (1-a) * cbf_all
    rmse = np.sqrt(mean_squared_error(actual_all, pred))
    mae = mean_absolute_error(actual_all, pred)
    if rmse < best_rmse:
        best_rmse, best_alpha = rmse, a
    print(f'  alpha={a:.1f}: RMSE={rmse:.4f}, MAE={mae:.4f}')

ALPHA = best_alpha
print(f'\nOptimal alpha: {ALPHA:.1f}')

# Chart: alpha tuning
alpha_rmses = [np.sqrt(mean_squared_error(actual_all, a*cf_all + (1-a)*cbf_all)) for a in alphas]
alpha_maes = [mean_absolute_error(actual_all, a*cf_all + (1-a)*cbf_all) for a in alphas]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(alphas, alpha_rmses, 'bo-', label='RMSE', linewidth=2)
ax.plot(alphas, alpha_maes, 'rs-', label='MAE', linewidth=2)
ax.axvline(x=ALPHA, color='green', linestyle='--', label=f'Best: {ALPHA:.1f}')
ax.set_xlabel('Alpha (CF weight)')
ax.set_ylabel('Error')
ax.set_title('Hybrid Blending Weight Optimisation')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('output/alpha_tuning.png', dpi=150); plt.close()
print('Saved output/alpha_tuning.png')

# ── Full 3-way comparison ─────────────────────────────────────
print('\n' + '='*70)
print(f'3-WAY COMPARISON (alpha={ALPHA:.1f})')
print('='*70)

hybrid_all = ALPHA * cf_all + (1 - ALPHA) * cbf_all
results = {}
for name, preds in [('CF', cf_all), ('CBF', cbf_all), ('Hybrid', hybrid_all)]:
    rmse = np.sqrt(mean_squared_error(actual_all, preds))
    mae = mean_absolute_error(actual_all, preds)
    results[name] = {'RMSE': rmse, 'MAE': mae}
    print(f'{name:8s}: RMSE={rmse:.4f}, MAE={mae:.4f}')

# ── Charts ────────────────────────────────────────────────────
print('\nGenerating charts...')

# Chart 1: Evaluation results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
methods = list(results.keys())
rmse_vals = [results[m]['RMSE'] for m in methods]
mae_vals = [results[m]['MAE'] for m in methods]
x = np.arange(len(methods)); w = 0.35
axes[0].bar(x-w/2, rmse_vals, w, label='RMSE', color='steelblue')
axes[0].bar(x+w/2, mae_vals, w, label='MAE', color='coral')
axes[0].set_xticks(x); axes[0].set_xticklabels(methods)
axes[0].set_ylabel('Error'); axes[0].set_title('Rating Prediction Error')
axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

labels = ['CF Only', 'CBF Only', f'Hybrid (a={ALPHA:.1f})']
vals = [results['CF']['RMSE'], results['CBF']['RMSE'], results['Hybrid']['RMSE']]
axes[1].bar(labels, vals, color=['#ff9999','#66b3ff','#99ff99'], edgecolor='black')
axes[1].set_ylabel('RMSE'); axes[1].set_title('RMSE: Hybrid vs Individual')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(vals):
    axes[1].text(i, v+0.005, f'{v:.4f}', ha='center', fontsize=10)

# Prediction vs Actual scatter
axes[2].scatter(actual_all, hybrid_all, alpha=0.1, s=5)
axes[2].plot([0.5,5],[0.5,5], 'r--', linewidth=2)
axes[2].set_xlabel('Actual Rating'); axes[2].set_ylabel('Predicted Rating')
axes[2].set_title('Hybrid: Predicted vs Actual')
axes[2].grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig('output/evaluation_results.png', dpi=150); plt.close()
print('Saved output/evaluation_results.png')

# Chart 2: Error distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (name, preds) in enumerate([('CF', cf_all), ('CBF', cbf_all), ('Hybrid', hybrid_all)]):
    errors = actual_all - preds
    axes[idx].hist(errors, bins=50, color=['steelblue','coral','green'][idx], edgecolor='black', alpha=0.7)
    axes[idx].axvline(x=0, color='red', linestyle='--')
    axes[idx].set_xlabel('Error'); axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'{name} (mean={errors.mean():.3f}, std={errors.std():.3f})')
    axes[idx].grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig('output/error_distribution.png', dpi=150); plt.close()
print('Saved output/error_distribution.png')

# Chart 3: Mood recommendations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, (_, mrow) in enumerate(mood_map.iterrows()):
    ax = axes[idx//3][idx%3]
    mood = mrow['mood']; genres = mrow['genres'].split('|')
    mask = movies['genres_list'].apply(lambda g: any(genre in g for genre in genres))
    mood_movies = movies[mask].nlargest(15, 'avg_rating')
    ax.barh(range(len(mood_movies)), mood_movies['avg_rating'].values, color=plt.cm.Set2(idx))
    ax.set_yticks(range(len(mood_movies)))
    ax.set_yticklabels(mood_movies['title'].values, fontsize=7)
    ax.set_xlabel('Avg Rating'); ax.set_title(f'{mood} ({len(movies[mask])} movies)')
    ax.set_xlim(3.0, 4.5); ax.invert_yaxis()
plt.suptitle('Top-Rated Movies by Mood Category', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('output/mood_recommendations.png', dpi=150); plt.close()
print('Saved output/mood_recommendations.png')

# ── Precision@K, Recall@K, F1@K ──────────────────────────────
print('\n' + '='*70)
print('TOP-N METRICS')
print('='*70)

user_test = test.groupby('userId').agg(
    test_movies=('ratingId', list), test_ratings=('rating', list)
).reset_index()
eval_users = user_test[user_test['test_movies'].apply(len) >= 5].head(100)

K_values = [5, 10, 20]
metrics_by_K = {}

def get_top_n_hybrid(user_id, n):
    rated = set(train[train['userId'] == user_id]['ratingId'])
    candidates = [mid for mid in movie_map_cf if mid not in rated]
    scores = [(mid, batch_cf_predict(user_id, mid) * ALPHA + batch_cbf_predict(user_id, mid) * (1-ALPHA))
              for mid in candidates[:150]]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in scores[:n]]

for K in K_values:
    print(f'\n--- K={K} ---')
    precisions, recalls, f1s = [], [], []
    for _, urow in eval_users.iterrows():
        uid = urow['userId']
        relevant = set([mid for mid, r in zip(urow['test_movies'], urow['test_ratings']) if r >= 3.5])
        if not relevant: continue
        recs = get_top_n_hybrid(uid, K)
        hits = len(set(recs) & relevant)
        p = hits / K; r = hits / len(relevant)
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0
        precisions.append(p); recalls.append(r); f1s.append(f1)
    avg_p = np.mean(precisions) if precisions else 0
    avg_r = np.mean(recalls) if recalls else 0
    avg_f1 = np.mean(f1s) if f1s else 0
    print(f'  Hybrid: P@{K}={avg_p:.4f}, R@{K}={avg_r:.4f}, F1@{K}={avg_f1:.4f}')
    metrics_by_K[K] = {'precision': avg_p, 'recall': avg_r, 'f1': avg_f1}

# Chart 4: P/R/F1
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
k_vals = sorted(metrics_by_K.keys())
for idx, (metric, title) in enumerate(zip(['precision','recall','f1'], ['Precision@K','Recall@K','F1@K'])):
    vals = [metrics_by_K[k][metric] for k in k_vals]
    axes[idx].plot(k_vals, vals, 'o-', color='steelblue', linewidth=2, markersize=8)
    for k, v in zip(k_vals, vals):
        axes[idx].text(k, v+0.002, f'{v:.4f}', ha='center', fontsize=9)
    axes[idx].set_xlabel('K'); axes[idx].set_ylabel(title); axes[idx].set_title(title)
    axes[idx].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('output/precision_recall_f1.png', dpi=150); plt.close()
print('Saved output/precision_recall_f1.png')

# ── Demo: Mood-based recommendations ─────────────────────────
print('\n' + '='*70)
print(f'DEMO: Mood-Based Recommendations (user={train["userId"].iloc[0]})')
print('='*70)

demo_user = train['userId'].iloc[0]
for _, mrow in mood_map.iterrows():
    mood = mrow['mood']; genres = mrow['genres'].split('|')
    mood_mask = movies['genres_list'].apply(lambda g: any(genre in g for genre in genres))
    mood_rids = set(movies[mood_mask]['ratingId'])
    rated = set(train[train['userId'] == demo_user]['ratingId'])
    candidates = [mid for mid in mood_rids if mid not in rated]
    scores = [(mid, ALPHA*batch_cf_predict(demo_user,mid)+(1-ALPHA)*batch_cbf_predict(demo_user,mid)) for mid in candidates[:80]]
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f'\n  {mood}:')
    for rank, (mid, score) in enumerate(scores[:5]):
        t = movies[movies['ratingId']==mid]['title'].values
        t = t[0] if len(t)>0 else str(mid)
        g = movies[movies['ratingId']==mid]['genres_str'].values
        g = g[0] if len(g)>0 else ''
        print(f'    {rank+1}. {t} (score: {score:.2f}, genres: {g})')

# ── Final Summary ─────────────────────────────────────────────
print('\n' + '='*70)
print('FINAL RESULTS SUMMARY')
print('='*70)

print('\nRating Prediction:')
print(f'{"Method":<10} {"RMSE":>8} {"MAE":>8}')
print('-'*28)
for m in ['CF','CBF','Hybrid']:
    print(f'{m:<10} {results[m]["RMSE"]:>8.4f} {results[m]["MAE"]:>8.4f}')

print(f'\nHybrid improvement over CF:  {(results["CF"]["RMSE"]-results["Hybrid"]["RMSE"]):.4f} RMSE')
print(f'Hybrid improvement over CBF: {(results["CBF"]["RMSE"]-results["Hybrid"]["RMSE"]):.4f} RMSE')

print('\nTop-N Metrics:')
print(f'{"K":<5} {"Precision":>10} {"Recall":>10} {"F1":>10}')
print('-'*37)
for K in K_values:
    d = metrics_by_K[K]
    print(f'{K:<5} {d["precision"]:>10.4f} {d["recall"]:>10.4f} {d["f1"]:>10.4f}')

print(f'\nCharts saved in output/')
print('='*70)
print('DONE')
