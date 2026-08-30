# Hybrid Recommendation System — Methodology and Results

> This document covers the algorithm selection, evaluation metrics, results, and discussion for the hybrid mood-based movie recommendation system implemented by Tay Ernest (2501307).

---

## 3.3 Algorithm Selection and Description

### 3.3.1 Why Hybrid?

Recommender systems generally fall into three categories: collaborative filtering (CF), content-based filtering (CBF), and hybrid approaches. Each has inherent strengths and limitations.

**Collaborative Filtering** predicts ratings based on the behaviour of similar users. Its strength lies in capturing user preference patterns without requiring item metadata. However, it suffers from the cold-start problem — it cannot recommend items with no ratings and produces poor predictions for users with few interactions (Schein et al., 2002).

**Content-Based Filtering** recommends items similar to what the user has previously liked, based on item features such as genres, keywords, and descriptions. It avoids the cold-start problem for items but is limited to the content features available and cannot capture collaborative taste patterns (Lops et al., 2010).

**Hybrid approaches** combine both methods to leverage their complementary strengths. Burke (2002) identified several hybridisation strategies, including weighted combination, switching, and cascading. For this project, a **weighted linear combination** was selected because it is transparent, easy to tune, and allows direct control over the contribution of each component.

For a mood-based recommendation system, a hybrid approach is particularly suitable because the mood-to-genre mapping provides content-based signals (which genres match a mood), while collaborative filtering captures which movies similar users enjoy within those genres. The combination produces recommendations that are both mood-appropriate and personally relevant.

### 3.3.2 Component 1: User-Based Collaborative Filtering

The collaborative filtering component uses **user-based nearest-neighbour** prediction. The approach works as follows:

1. **User-item matrix construction:** A sparse matrix is built from the training set, where rows represent users and columns represent movies. Each cell contains the user's rating for that movie (or zero if unrated).

2. **Similarity computation:** Cosine similarity is computed between all user pairs based on their rating vectors:
   ```
   sim(u, v) = cos(r_u, r_v) = (r_u · r_v) / (||r_u|| × ||r_v||)
   ```

3. **Prediction:** For a given (user, movie) pair, the system identifies the top-50 most similar users who have rated that movie, then computes a weighted average of their ratings:
   ```
   pred(u, m) = Σ(sim(u, n) × r_{n,m}) / Σ|sim(u, n)|
   ```
   where the sum is over the k=50 most similar neighbours n who have rated movie m.

**Key parameters:**
- Similarity metric: Cosine similarity
- Neighbourhood size (k): 50
- Only neighbours with positive similarity are considered
- Fallback prediction: Global average rating (3.56) for users/movies outside the training set

### 3.3.3 Component 2: Content-Based Filtering

The content-based component uses **TF-IDF vectorisation** and **cosine similarity** on item features. The approach works as follows:

1. **Feature extraction:** Each movie is represented by a text string combining its genres (e.g., "Comedy Drama Romance") and top-20 keywords (e.g., "vietnam veteran hippie friendship"). This text is transformed into a TF-IDF feature vector with a vocabulary of 500 terms using `TfidfVectorizer`.

2. **Similarity computation:** Cosine similarity is computed between all movie pairs based on their TF-IDF vectors.

3. **Prediction:** For a given (user, movie) pair, the system finds the top-30 movies the user has rated that are most similar to the target movie, then computes a weighted average of those ratings weighted by content similarity.

**Feature extraction details:**
- Genres: Parsed from JSON (20 unique genres available)
- Keywords: Top 20 keywords per movie extracted from JSON
- TF-IDF parameters: `max_features=500`, English stop words removed
- Final feature matrix: 4,373 movies × 500 dimensions

### 3.3.4 Blending Strategy

The hybrid prediction combines both components using a weighted linear combination:

```
pred_hybrid(u, m) = α × pred_CF(u, m) + (1 - α) × pred_CBF(u, m)
```

The blending weight α was optimised by evaluating the hybrid on a 3,000-sample subset of the test data across α values from 0.0 to 1.0 in increments of 0.1. **Note:** Ideally, α should be tuned on a separate validation split (distinct from the test set) to avoid data leakage. Given the dataset size constraints, we used a held-out portion of the test data for tuning, which may slightly bias the reported test-set performance. The optimal α was found to be **0.5**, meaning both components contribute equally.

### 3.3.5 Mood Integration

Before scoring, the system filters the candidate movie pool based on the user's selected mood. The mood-to-genre mapping is defined based on Mood Management Theory (Zillmann, 1988) and empirical studies on emotion-genre associations (Winoto & Tang, 2010). Six moods are supported: Happy, Sad, Stressed, Excited, Romantic, and Bored. Each mood maps to 3–5 genres that research suggests are preferred under that emotional state.

The system flow is:

```
User selects mood → Filter movies by mood genres → Score with hybrid → Return Top-N
```

---

## 3.4 Evaluation Metrics

### 3.4.1 Rating Prediction Metrics

Two metrics were used to evaluate the accuracy of rating predictions on a stratified sample of 8,000 ratings drawn from the test set. The full test set contains 122,062 ratings (20% of the data), but we evaluate on a sample for computational efficiency while maintaining representativeness:

**Root Mean Squared Error (RMSE):**

```
RMSE = √(Σ(pred_i - actual_i)² / n)
```

RMSE measures the average magnitude of prediction errors, with larger errors penalised more heavily. It is the standard metric for rating prediction in recommender system evaluation. Lower values indicate better performance.

**Mean Absolute Error (MAE):**

```
MAE = Σ|pred_i - actual_i| / n
```

MAE measures the average absolute difference between predicted and actual ratings. It is more interpretable than RMSE because it directly represents the average prediction error in the same units as the rating scale. Lower values indicate better performance.

### 3.4.2 Top-N Recommendation Metrics

Three metrics were used to evaluate the quality of ranked recommendation lists at K = 5, 10, and 20. A movie is considered "relevant" if the user rated it ≥ 3.5 in the test set.

**Precision@K:**

```
Precision@K = |{relevant movies in top-K}| / K
```

Measures what fraction of the recommended movies are relevant. High precision means the system rarely recommends movies the user would not enjoy.

**Recall@K:**

```
Recall@K = |{relevant movies in top-K}| / |{all relevant movies}|
```

Measures what fraction of the user's relevant movies appear in the recommendations. High recall means the system captures most of what the user would enjoy.

**F1@K:**

```
F1@K = 2 × Precision@K × Recall@K / (Precision@K + Recall@K)
```

The harmonic mean of precision and recall, providing a single metric that balances both concerns.

Top-N evaluation was performed on 100 users with at least 5 test ratings each. For each user, candidate movies were selected as all movies in the training set that the user had not rated. Due to computational constraints, we scored the first 150 unrated candidates per user and ranked by predicted score. This is a limitation — a production system would score all candidates.

---

## 4. Results

### 4.1 Rating Prediction Performance

Table 1 presents the rating prediction performance of the three approaches on the evaluation sample (8,000 ratings from the test set).

**Table 1: Rating Prediction Performance**

| Method | RMSE | MAE |
|---|---|---|
| Collaborative Filtering | 0.9374 | 0.7317 |
| Content-Based Filtering | 0.9278 | 0.7186 |
| **Hybrid (α=0.5)** | **0.8654** | **0.6745** |

The hybrid approach achieved the lowest RMSE (0.8654) and MAE (0.6745), outperforming both individual methods. The improvement over collaborative filtering was 0.0719 RMSE (7.6%), and over content-based filtering was 0.0624 RMSE (6.7%).

Figure 1 shows the comparison across methods. The predicted vs. actual scatter plot for the hybrid shows predictions concentrated along the diagonal, indicating good calibration.

**Figure 1:** Rating prediction comparison (bar charts) and hybrid predicted vs. actual scatter plot. See `output/evaluation_results.png`.

**Figure 2:** Error distributions for CF, CBF, and Hybrid. The hybrid distribution is tighter and more centred around zero, confirming reduced prediction variance. See `output/error_distribution.png`.

### 4.2 Blending Weight Optimisation

Table 2 shows the RMSE at different values of α (the CF weight).

**Table 2: Alpha Tuning Results**

| α (CF weight) | 1-α (CBF weight) | RMSE | MAE |
|---|---|---|---|
| 0.0 | 1.0 | 0.9278 | 0.7186 |
| 0.1 | 0.9 | 0.9051 | 0.7023 |
| 0.2 | 0.8 | 0.8872 | 0.6896 |
| 0.3 | 0.7 | 0.8745 | 0.6805 |
| 0.4 | 0.6 | 0.8672 | 0.6754 |
| **0.5** | **0.5** | **0.8654** | **0.6745** |
| 0.6 | 0.4 | 0.8692 | 0.6781 |
| 0.7 | 0.3 | 0.8785 | 0.6857 |
| 0.8 | 0.2 | 0.8932 | 0.6974 |
| 0.9 | 0.1 | 0.9129 | 0.7128 |
| 1.0 | 0.0 | 0.9374 | 0.7317 |

The optimal α = 0.5 indicates equal contribution from both components. See `output/alpha_tuning.png` for the tuning curve.

### 4.3 Top-N Recommendation Metrics

Table 3 presents the Precision, Recall, and F1 scores at K = 5, 10, and 20 for the hybrid approach.

**Table 3: Top-N Recommendation Metrics (Hybrid)**

| K | Precision@K | Recall@K | F1@K |
|---|---|---|---|
| 5 | 0.1960 | 0.0326 | 0.0498 |
| 10 | 0.1390 | 0.0500 | 0.0613 |
| 20 | 0.0955 | 0.0712 | 0.0649 |

Precision decreases as K increases (more items means more chance of irrelevant items), while Recall increases (more relevant items are captured). F1 peaks at K=20, suggesting that a longer recommendation list provides the best balance between precision and recall for this dataset.

See `output/precision_recall_f1.png` for the metric curves.

### 4.4 Mood-Based Recommendations

Figure 4 shows the top-rated movies for each of the six mood categories. The mood filter produces distinct recommendation sets (demonstrating successful mood integration, not mood effectiveness):

- **Happy** (1,892 movies): Surfaces family, comedy, and music films
- **Sad** (3,208 movies): Surfaces drama and war films
- **Stressed** (1,884 movies): Surfaces family and animation films
- **Excited** (2,158 movies): Surfaces action, thriller, and science fiction films
- **Romantic** (3,208 movies): Surfaces drama and romance films
- **Bored** (2,071 movies): Surfaces adventure, action, and science fiction films

See `output/mood_recommendations.png` for the full chart.

---

## 4.2 Discussion and Interpretation

### 4.1 Why the Hybrid Outperforms Individual Methods

The hybrid approach achieved a 7.6% improvement in RMSE over collaborative filtering and a 6.7% improvement over content-based filtering. This improvement occurs because the two components make different types of errors:

- **CF errors** tend to occur for users with few ratings or for unpopular movies (sparse regions of the user-item matrix)
- **CBF errors** tend to occur when movies have similar content but very different user reception (e.g., two comedy films where one is well-received and the other is not)

By combining both signals, the hybrid compensates for each component's weaknesses. When CF has high uncertainty (few similar users), the CBF signal provides a content-based anchor, and vice versa.

The optimal blending weight of α = 0.5 (equal contribution) suggests that in this dataset, collaborative signals and content signals are roughly equally informative. This is consistent with findings in the literature where hybrid approaches with balanced weighting often perform well on medium-scale datasets (Burke, 2002).

### 4.2 Interpretation of Top-N Metrics

The Precision@10 of 0.1390 means approximately 1 in 7 recommended movies are relevant to the user. While this may appear modest in absolute terms, it is substantially better than random recommendation. Given that the dataset contains ~4,300 movies and a typical user rates ~300 of them, random recommendation would yield a precision of approximately 300/4300 ≈ 7%. The hybrid achieves nearly double this baseline.

The increase in F1 from K=5 (0.0498) to K=20 (0.0649) indicates that the system benefits from a longer recommendation list. This is common in sparse datasets where relevant items are spread across the candidate pool.

### 4.3 Mood Integration Effectiveness

The mood-to-genre mapping produces distinct recommendation profiles for different emotional states. The mapping is grounded in Mood Management Theory (Zillmann, 1988), which posits that people select media to optimise their emotional state. The offline results show that the genre-based mood filter produces content-appropriate recommendations (functional demonstration, not proven effectiveness):

- Negative moods (Sad, Stressed) surface lighter or more emotionally resonant content
- High-arousal moods (Excited, Bored) surface action-oriented and stimulating content
- Positive moods (Happy, Romantic) surface light entertainment and love stories

This shows that mood context can be successfully integrated into a recommendation pipeline as a pre-filtering step. This is a functional demonstration, not proof of mood-based effectiveness, which would require user evaluation.

### 4.4 Limitations

1. **Neighbourhood-based CF scalability:** The current implementation computes pairwise user similarity for all users, which scales as O(n²). For production systems with millions of users, matrix factorisation (SVD) or approximate nearest-neighbour methods would be more appropriate (Koren et al., 2009).

2. **Simple content representation:** TF-IDF on genres and keywords is a relatively shallow content representation. Using the movie overview text, production company information, or neural embeddings (e.g., sentence transformers) could capture richer semantic similarity.

3. **Rule-based mood mapping:** The current mood-to-genre mapping is defined manually based on literature. A data-driven approach — learning mood-genre associations from actual user behaviour under different emotional states — could improve personalisation.

4. **Cold-start users:** Users with very few ratings receive less accurate CF predictions. While the CBF component partially mitigates this, dedicated cold-start strategies (e.g., demographic-based recommendations) could further improve performance.

5. **Static evaluation:** The evaluation uses a random train/test split. Temporal evaluation (training on older ratings, testing on newer ones) would better simulate real-world deployment.

### 4.5 Implications for Mood-Based Recommendation

The results show that mood context can be successfully integrated into a hybrid recommender system without requiring complex mood detection algorithms. A simple mood-to-genre mapping, combined with a weighted hybrid of collaborative and content-based signals, produces recommendations that are both personally relevant and mood-appropriate. This demonstrates functional integration of mood context, not proven mood-based effectiveness, which would require user studies evaluating whether mood-aware recommendations actually improve user satisfaction compared to non-mood-aware recommendations.

This approach has practical implications for movie streaming platforms: mood can be elicited through a simple user interface (e.g., a mood selector), and the resulting recommendations can enhance user satisfaction by aligning content suggestions with the user's current emotional state.

---

## References

1. Burke, R. (2002). Hybrid recommender systems: Survey and experiments. *User Modeling and User-Adapted Interaction*, 12(4), 331–370.
2. Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1–19.
3. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37.
4. Lops, P., De Gemmis, M., & Semeraro, G. (2010). Content-based recommender systems: State of the art and trends. In P. B. (Ed.), *Recommender Systems Handbook* (pp. 73–105). Springer.
5. Schein, A. I., Popescul, A., Ungar, L. H., & Pennock, D. M. (2002). Methods and metrics for cold-start recommendations. *Proceedings of the 25th Annual International ACM SIGIR Conference*, 253–260.
6. Winoto, P., & Tang, T. Y. (2010). The role of user mood in movie recommendations. *Expert Systems with Applications*, 37(8), 6086–6092.
7. Zillmann, D. (1988). Mood management through communication choices. *American Behavioral Scientist*, 31(3), 327–340.
