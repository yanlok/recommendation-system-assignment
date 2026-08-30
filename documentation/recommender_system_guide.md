# Mood-Based Movie Recommendation System — Viva & Q&A Guide

> **Purpose:** This document is a study guide to help each group member explain the system during the Week 12–14 demonstration and Q&A session. Read this alongside your own sections of the report. Every question below is the kind a lecturer would ask to test whether you truly understand what you built.

---

## 1. High-Level Overview

### Q: What does your system do in one sentence?

**A:** The system recommends movies to a user based on their current mood. The user selects one of six moods (Happy, Sad, Stressed, Excited, Romantic, Bored), and the system filters movies whose genres match that mood, then ranks them using one of three recommendation algorithms.

### Q: Why is this different from a normal recommender system?

**A:** Most recommender systems recommend based on a user's historical preferences — what they rated highly in the past. Our system adds a contextual layer: the user's current emotional state. This means the same user can get different recommendations depending on how they feel right now. A user who usually watches action films might get comedy recommendations when they select "Stressed."

---

## 2. The Three Algorithms

The assignment requires each group member to implement a different recommender solution. Our group chose:

| Member | Algorithm | Core Idea |
|--------|-----------|-----------|
| Chay Qian | Collaborative Filtering (CF) | "Find users like you and recommend what they liked" |
| Wong Jin Yu | Content-Based Filtering (CBF) | "Find movies similar to what you already liked" |
| Tay Ernest | Hybrid (Weighted Combination) | "Combine both signals for a more robust recommendation" |

### Q (for CF member): How does collaborative filtering work in your system?

**A:** Collaborative filtering works by finding users who have similar rating patterns to the target user. The process is:

1. **Build a user-item matrix:** Every user's ratings are stored in a matrix where rows are users, columns are movies, and values are the ratings (0 means unrated).

2. **Compute user similarity:** We use cosine similarity on mean-centred ratings. Mean-centring subtracts each user's average rating from all their ratings, so the similarity focuses on the *pattern* of preferences rather than whether someone rates generously or harshly. Two users are similar if they rate the same movies similarly relative to their personal averages.

3. **Find top-K neighbours:** For the target user, we find the 50 most similar users who have positive similarity scores.

4. **Predict a rating:** For each candidate movie, we take a weighted average of the ratings from those similar users. The weights are the similarity scores — more similar users have more influence:

   ```
   Predicted rating = Σ(similarity × rating) / Σ|similarity|
   ```

5. **Rank and recommend:** Movies with the highest predicted ratings are recommended. Movies the user already rated are excluded.

### Q (for CBF member): How does content-based filtering work?

**A:** Content-based filtering recommends movies based on their content features — genres, keywords, overview, cast, and language. The process is:

1. **Feature extraction with TF-IDF:** We combine textual movie features (genres, keywords, overview, cast, original language) into a single text representation for each movie. TF-IDF (Term Frequency–Inverse Document Frequency) converts this text into numerical vectors. TF-IDF assigns higher weights to terms that are distinctive to a particular movie while reducing the importance of common words that appear in many movies.

2. **Compute item similarity:** Cosine similarity is computed between all movie vectors. This produces a 4,373 × 4,373 item-item similarity matrix. Two movies with similar genres, keywords, and cast will have high similarity.

3. **Predict a rating:** For a target user and a candidate movie, we look at the user's previously rated movies. We take a weighted average of the user's ratings on their most similar rated movies:

   ```
   Predicted rating = Σ(similarity × user's rating) / Σ|similarity|
   ```

   Only movies with positive similarity to the candidate are included. This is clipped to the valid range [0.5, 5.0].

4. **Rank and recommend:** Movies with the highest predicted ratings are recommended.

### Q (for Hybrid member): Why use a hybrid approach? How does the weighting work?

**A:** A hybrid approach combines collaborative filtering and content-based filtering to get the strengths of both while reducing their individual weaknesses.

**Why hybrid?**
- CF struggles with the cold-start problem (new users with few ratings).
- CBF can only recommend movies similar to what you already watched — it cannot discover new preferences from similar users.
- A hybrid can balance both signals.

**How the weighting works:**

We use a weighted linear combination:

```
Hybrid Score = α × (CF Prediction) + (1 − α) × (CBF Prediction)
```

where α controls how much weight each component gets.

**How α was chosen:** We evaluated α values from 0.0 to 1.0 in increments of 0.1. At α = 0.5, the hybrid achieved the lowest RMSE (0.8741), so equal weighting (50% CF + 50% CBF) was selected. This is shown in the alpha tuning chart in the evaluation results.

**CF in the hybrid uses raw ratings cosine similarity** (not mean-centred), while **CBF uses a 500-feature TF-IDF** (not 20k like the standalone CBF). Both use global-mean fallback when there is no signal, so the hybrid never produces NaN predictions.

---

## 3. The Mood Filtering Mechanism

### Q: How does mood affect the recommendations?

**A:** Mood acts as a **pre-filter** before the recommendation algorithm runs. Here is the pipeline:

1. User selects a mood (e.g., "Excited").
2. The mood-to-genre mapping associates "Excited" with genres: Action, Adventure, Thriller, Science Fiction.
3. The system finds all movies that have at least one of these genres.
4. Movies the user already rated are removed from the candidate pool.
5. The selected recommendation algorithm (CF, CBF, or Hybrid) scores the remaining candidates.
6. The top-N movies are returned.

### Q: Where did the mood-to-genre mapping come from?

**A:** The mapping was informed by Mood Management Theory (Winoto & Tang, 2010) and the literature on emotion-aware recommender systems. The idea is that people in different emotional states seek different types of entertainment:

| Mood | Associated Genres | Rationale |
|------|-------------------|-----------|
| Happy | Comedy, Animation, Family, Music | Light, enjoyable content to maintain positive mood |
| Sad | Drama, Romance, Comedy | Comfort or emotional resonance |
| Stressed | Comedy, Animation, Family, Documentary | Relaxing, low-stakes viewing |
| Excited | Action, Adventure, Thriller, Science Fiction | High-energy, stimulating content |
| Romantic | Romance, Drama, Comedy | Emotionally engaging, relationship-focused |
| Bored | Adventure, Action, Science Fiction, Mystery, Horror | Stimulating, immersive content to break boredom |

### Q: What happens if a user has very few ratings (cold start)?

**A:** Users with fewer than 20 ratings are flagged as "sparse" users. For these users, the system falls back to a **popularity-based ranking** within the mood-filtered candidates. The popularity score uses a Bayesian weighted average:

```
Score = (v / (v + m)) × R + (m / (v + m)) × C
```

where:
- `v` = number of ratings the movie has
- `m` = prior strength (set to 100)
- `R` = movie's average rating
- `C` = global average rating across all movies

This prevents a movie with a single 5.0 rating from outranking a movie with a 4.3 average from 1,000 ratings.

---

## 4. Data and Preprocessing

### Q: What dataset did you use and why?

**A:** We used the TMDB 5000 Movie Dataset with Ratings from Kaggle (Soni, 2024). It contains:
- 4,602 movies with metadata (genres, keywords, overview, cast, etc.)
- 17.2 million user ratings from 162,532 users
- Ratings from 0.5 to 5.0

We chose it because it provides both item content (for content-based filtering) and user-item interactions (for collaborative filtering), which is exactly what our three algorithms need.

### Q: How did you handle the data size?

**A:** The full 17.2 million ratings would be too slow to work with interactively. We performed **user-based random subsampling**: we randomly selected 2,000 users and kept all their ratings. This reduced the dataset to 610,307 ratings while preserving complete rating histories for each sampled user. The user-item matrix has a density of about 7% (sparsity of 93%), which is typical for real-world recommender systems.

### Q: What preprocessing steps did you do?

**A:**
1. Removed 7 movies with no user rating records (cold-start items that can't participate in CF).
2. Resolved 4 duplicated tmdbId values by reassigning 315 ratings to the retained movie records.
3. Removed 11 movies with empty genre lists (needed for mood-to-genre mapping).
4. Parsed JSON-formatted genres and keywords into structured lists.
5. Computed per-movie statistics: average rating, total ratings, rating standard deviation.

---

## 5. Evaluation

### Q: How did you evaluate the system?

**A:** We used two categories of metrics, applied with an 80/20 train-test split (random_state=42):

**Rating Prediction Accuracy:**
- **RMSE** (Root Mean Squared Error): Measures how far predicted ratings are from actual ratings. Lower is better. Sensitive to large errors.
- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual ratings. Lower is better. Easier to interpret.

**Top-N Recommendation Quality:**
- **Precision@K**: Of the K movies recommended, how many did the user actually like? (Relevance threshold: rating ≥ 3.5)
- **Recall@K**: Of all the movies the user liked, how many appeared in the top-K?
- **F1@K**: Harmonic mean of Precision and Recall — a balanced measure.

We evaluated at K = 5, 10, and 20, using 100 users who had at least 5 ratings in the test set.

### Q: What were the results?

**A:**

**Rating Prediction (8,000 test ratings):**

| Method | RMSE | MAE |
|--------|------|-----|
| Collaborative Filtering | 0.9332 | 0.7289 |
| Content-Based Filtering | 0.9542 | 0.7404 |
| **Hybrid (α=0.5)** | **0.8741** | **0.6818** |

The hybrid achieves the lowest error — roughly 6-7% improvement over both individual methods.

**Top-N Quality (Hybrid, 100 users):**

| K | Precision | Recall | F1 |
|---|-----------|--------|----|
| 5 | 0.1260 | 0.0191 | 0.0308 |
| 10 | 0.1180 | 0.0323 | 0.0463 |
| 20 | 0.1135 | 0.0712 | 0.0756 |

Precision decreases as K increases (more slots = more dilution), while Recall increases (more of the user's liked movies are captured).

### Q: Why are the Precision/Recall values relatively low?

**A:** This is expected in a realistic setting with high data sparsity (93%). The evaluation pipeline scores *all* unrated movies as candidates (thousands of them), making it very hard to place relevant movies in the top-K. In practice, the mood pre-filter already narrows the pool significantly, which improves effective precision. The numbers are also consistent with what other academic studies report for similar-sized datasets.

---

## 6. The Streamlit Prototype

### Q: Walk me through the user interface.

**A:** The app has three tabs:

1. **🎬 Movie Recommendations** — The main tab. The user selects who they are (3 demo users with different sparsity levels), how they feel, how many movies to recommend, and which algorithm to use. Clicking "Recommend Movies" shows ranked movie cards with scores, genres, and a brief explanation of why each movie was recommended.

2. **⚖️ Algorithm Comparison** — Runs the same user + mood through all three algorithms side by side, so you can see how CF, CBF, and Hybrid differ.

3. **📊 Evaluation Results** — Shows pre-computed evaluation charts and metrics (generated offline by `scripts/run_evaluation.py`). This tab does not recompute results on user request.

### Q: Why only 3 demo users?

**A:** The demo users were chosen to represent three different sparsity scenarios:
- **User A (1,546 ratings):** Heavy rater — strong personalization, all algorithms work well.
- **User B (198 ratings):** Medium rater — typical user, moderate personalization.
- **User C (18 ratings):** Sparse rater — triggers the cold-start fallback (popularity-based).

This demonstrates the system's behaviour across the sparsity spectrum.

---

## 7. Architecture and Design Decisions

### Q: Why did you separate model preparation from the app?

**A:** The precomputation approach was chosen for performance. Computing TF-IDF vectors, building user-item matrices, and calculating cosine similarities are expensive operations that take minutes. By running `scripts/prepare_models.py` once offline, the Streamlit app only needs to load precomputed `.npz` and `.npy` files, which takes seconds. This makes the prototype responsive for live demonstration.

### Q: Why use sparse matrices?

**A:** The user-item matrix has 1,970 users × 4,369 movies = 8.6 million cells, but only 610,307 are filled (7% density). Storing this as a dense matrix would waste memory. A CSR (Compressed Sparse Row) matrix only stores the non-zero values, reducing memory from ~34 MB (dense float32) to a few MB.

### Q: Why cosine similarity and not Euclidean distance?

**A:** Cosine similarity measures the *angle* between two vectors, not their magnitude. This is important because two users might rate the same movies similarly but at different scales (one rates everything 3–5, the other rates 1–4). Cosine similarity captures the pattern alignment regardless of rating scale. Euclidean distance would be dominated by the magnitude difference.

### Q: What is mean-centring and why use it for CF?

**A:** Mean-centring subtracts each user's average rating from all their ratings before computing similarity. Without it, two users who rate everything high (e.g., both average 4.5) would appear similar even if they like completely different movies. After mean-centring, what matters is the *pattern* of above-average and below-average ratings, not the absolute level.

---

## 8. Limitations and Weaknesses (Be Ready to Discuss These)

### Q: What are the main limitations of your system?

**A:**
1. **Manual mood selection:** The user must self-report their mood, which may not always be accurate or fit neatly into six categories.
2. **Fixed mood-genre mapping:** The same mapping applies to all users, but different users may prefer different genres for the same mood (one sad person wants comedy, another wants drama).
3. **No user-mood learning:** The system does not learn from which recommendations users actually select under each mood. It always uses the same mood-to-genre mapping.
4. **Cold-start problem:** New users with very few ratings get popularity-based recommendations, which are less personalized.
5. **Static model:** The similarity matrices are precomputed and not updated as new ratings come in.

### Q: How could you improve the system?

**A:**
- **Personalized mood profiles:** Record which movies users select under each mood and learn per-user mood-to-preference mappings.
- **Automatic mood detection:** Use NLP sentiment analysis on user text input to infer mood, or use facial expression recognition (with consent).
- **More mood categories:** Allow multiple moods or intensity levels instead of a single fixed selection.
- **Dynamic updates:** Retrain similarity matrices periodically as new ratings arrive.
- **Explainability:** Add more detailed explanations for why each movie was recommended.

---
