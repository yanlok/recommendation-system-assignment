# Mood-Based Movie Recommendation System Prototype Plan

## 1. Purpose

This document is the execution plan for building the prototype of the **Mood-Based Movie Recommendation System** for the Artificial Intelligence assignment.

The prototype should be:

- simple to implement;
- easy for users to understand;
- visually clean and user friendly;
- able to demonstrate the AI recommendation logic clearly;
- reliable during the lecturer demonstration;
- easy for each group member to explain;
- aligned with the assignment documentation and proposed system design;
- submitted together with readable source code.

The recommended implementation platform is **Python + Streamlit** because the existing recommendation algorithms are already implemented in Python and Streamlit can provide a web-style interface without requiring a separate frontend framework.

---

# 2. Assignment Requirements to Keep in Mind

The prototype is not only a user interface. It must demonstrate a working AI application.

For the recommender-system topic, the assignment requires:

1. a real-life recommendation scenario;
2. a recommender-system solution from each group member;
3. implementation using Python or another relevant tool;
4. testing and evaluation of the recommender system;
5. suitable evaluation metrics such as Precision, Recall, F1, MSE or RMSE;
6. a working prototype with source code;
7. individual demonstration and Q&A.

The prototype assessment focuses on:

- **User Interface / Output**
- **Programming**
- **Degree of Completion**
- **System Implementation**
- **Presentation and On-the-Spot Coding**

Therefore, the prototype should not focus only on appearance. It must also demonstrate correct recommendation logic, validation, stable execution, and clear source-code organisation.

---

# 3. Prototype Goal

The system should allow a user to:

1. select their current mood;
2. select or enter a user profile;
3. select a recommendation algorithm if required for demonstration;
4. request movie recommendations;
5. receive a ranked list of suitable movies;
6. view useful information explaining each recommendation;
7. view evaluation results for the implemented algorithms.

The main user journey should remain short:

```text
Open Application
      ↓
Select User
      ↓
Select Mood
      ↓
Choose Number of Recommendations
      ↓
Click Recommend
      ↓
Mood-Based Candidate Filtering
      ↓
Recommendation Algorithm
      ↓
Rank Results
      ↓
Display Top-N Movies
```

For the hybrid model:

```text
Selected Mood
      ↓
Mood → Genre Mapping
      ↓
Filter Candidate Movies
      ↓
Collaborative Filtering Score
      +
Content-Based Filtering Score
      ↓
Weighted Hybrid Score
      ↓
Rank Movies
      ↓
Top-N Recommendations
```

---

# 4. Recommended Technology

## 4.1 Main Technology

Use:

```text
Python
+
Streamlit
+
Pandas
+
NumPy
+
Scikit-learn
```

Possible supporting libraries:

```text
joblib
pickle
matplotlib
```

Use additional libraries only if they are genuinely required.

## 4.2 Why Streamlit

Streamlit is suitable because:

- it is free;
- it is Python-based;
- it works directly with the existing recommender code;
- it can create a browser-based interface quickly;
- it supports tables, cards, charts, buttons, tabs and sidebars;
- the lecturer can run the program locally;
- the complete source code can be submitted;
- it reduces unnecessary frontend complexity.

Avoid introducing React, Node.js, databases or cloud services unless they are clearly needed.

---

# 5. Design Principle: Keep the Prototype Simple

The application should feel complete without becoming unnecessarily complicated.

The core principle is:

> Every feature should either improve recommendation quality, user experience, assignment assessment, or demonstration reliability.

Avoid adding features only because they look impressive.

Examples of unnecessary scope for the current assignment:

- user registration;
- password authentication;
- payment system;
- social-media integration;
- complicated database management;
- live chat;
- webcam emotion detection;
- complex admin dashboard;
- real-time TMDB API dependency.

These features may increase the risk of bugs without improving the main AI objective.

---

# 6. Recommended Application Structure

Use three main pages or tabs.

```text
1. Movie Recommendations
2. Algorithm Comparison
3. Evaluation Results
```

Optional fourth page:

```text
4. About the System
```

---

# 7. Page 1 — Movie Recommendations

This is the main page and should be the focus of the demonstration.

## 7.1 Header

Display:

```text
🎬 MoodFlix
Mood-Based Movie Recommendation System
```

Add a short description such as:

> Select your current mood and receive personalised movie recommendations.

Keep the introduction short.

---

## 7.2 User Selection

Provide a user selector.

Example:

```text
Select User
[ User 1024 ▼ ]
```

For demonstration purposes, a dropdown is safer than requiring the lecturer to type a valid ID.

If the user IDs are too numerous, provide:

- a searchable dropdown; or
- several prepared demo users.

Example:

```text
Demo User A
Demo User B
Demo User C
```

The actual user ID can still be shown for transparency.

### Important

Prepare several users with different rating histories before the demo.

Do not rely on randomly selecting a user during the presentation.

---

# 8. Mood Selection

The system currently supports six moods:

- 😀 Happy
- 😢 Sad
- 😰 Stressed
- 🔥 Excited
- ❤️ Romantic
- 🥱 Bored

The mood should be easy to select using:

- buttons;
- cards;
- radio buttons; or
- a select box.

The user should not need to understand the genre mapping.

For example:

```text
How are you feeling today?

😀 Happy
😢 Sad
😰 Stressed
🔥 Excited
❤️ Romantic
🥱 Bored
```

---

# 9. Mood-to-Genre Mapping

The current project uses the following mood mapping.

| Mood | Associated Genres |
|---|---|
| Happy | Comedy, Animation, Family, Music |
| Sad | Drama, Romance, Comedy |
| Stressed | Comedy, Animation, Family, Documentary |
| Excited | Action, Adventure, Thriller, Science Fiction |
| Romantic | Romance, Drama, Comedy |
| Bored | Adventure, Action, Science Fiction, Mystery, Horror |

This mapping should be stored separately in:

```text
data/processed/mood_genre_mapping.csv
```

Do not hard-code the same mapping in many files.

A single source should be used to reduce inconsistency.

---

# 10. Recommendation Settings

Keep settings limited.

Recommended options:

```text
Number of recommendations:
5
10
15
20
```

Default:

```text
10
```

Optional for the lecturer demonstration:

```text
Algorithm:
Collaborative Filtering
Content-Based Filtering
Hybrid
```

For the normal user experience, the system can default to the group's preferred or final recommendation method.

---

# 11. Recommendation Button

Use one obvious action:

```text
🎬 Recommend Movies
```

Do not automatically recompute expensive recommendations every time the user changes a small interface control.

The system should calculate recommendations when the user presses the button.

---

# 12. Recommendation Result Design

The output should contain more than movie titles.

Recommended information:

- Rank
- Movie title
- Genres
- Predicted rating / recommendation score
- Selected mood
- Match explanation

Example:

```text
#1 Inception

Genres:
Action • Science Fiction • Thriller

Predicted Rating:
4.42 / 5

Why recommended:
Matches your Excited mood and has a high predicted preference score.
```

Another compact layout:

| Rank | Movie | Genres | Predicted Rating |
|---:|---|---|---:|
| 1 | Inception | Action, Sci-Fi, Thriller | 4.42 |
| 2 | The Dark Knight | Action, Crime, Thriller | 4.35 |
| 3 | Guardians of the Galaxy | Action, Adventure, Sci-Fi | 4.29 |

---

# 13. Explainability

A successful recommender system should give users some idea of why an item was recommended.

The explanation does not need to be technically complicated.

Examples:

```text
Recommended because it matches your Excited mood.
```

```text
Recommended because you previously rated similar Action and Science Fiction movies highly.
```

```text
Recommended using both your rating history and movie-content similarity.
```

For the hybrid system:

```text
Mood Match:
Excited → Action / Adventure / Thriller / Science Fiction

Personalisation:
Collaborative filtering + content similarity
```

This improves user trust and also helps explain the system to the lecturer.

---

# 14. Avoid Recommending Movies the User Already Rated

A recommendation system should generally recommend unseen items.

Before ranking candidates:

```text
Candidate Movies
=
Mood-Matching Movies
-
Movies Already Rated by User
```

This is an important quality rule.

If already-seen movies are included during testing for a specific reason, clearly separate that evaluation process from the normal recommendation interface.

---

# 15. Recommendation Ranking

The final output should be ranked from strongest to weakest recommendation.

For the hybrid model:

```text
Hybrid Score
=
α × Collaborative Score
+
(1 - α) × Content-Based Score
```

Current project result:

```text
α = 0.5
```

Therefore:

```text
Hybrid Score
=
0.5 × Collaborative Score
+
0.5 × Content-Based Score
```

The highest scores should appear first.

---

# 16. Collaborative Filtering Component

The collaborative component currently uses:

- user-based collaborative filtering;
- cosine similarity;
- nearest neighbours;
- neighbourhood size `k = 50`;
- weighted-average rating prediction.

Required inputs:

```text
User
Movie
User Rating History
Other Users' Ratings
```

Output:

```text
Predicted Rating
```

Considerations:

- user must exist in the dataset;
- target movie must exist;
- similar users may not have rated the target item;
- sparse user profiles require fallback handling;
- already-rated items should be excluded from normal recommendations.

---

# 17. Content-Based Filtering Component

The content-based component currently uses:

- genres;
- movie keywords;
- TF-IDF;
- cosine similarity.

Current feature settings include:

```text
Genres
+
Top 20 Keywords

TF-IDF max_features = 500
```

Output:

```text
Content-Based Predicted Preference
```

Considerations:

- movies must have usable feature data;
- empty text should be handled;
- missing overview or optional metadata should not crash the program;
- TF-IDF should be built once rather than recalculated for every click.

---

# 18. Hybrid Recommendation Component

The hybrid system combines:

```text
Collaborative Preference
+
Content Similarity
+
Mood Filtering
```

Recommended implementation sequence:

```text
1. Validate user
2. Read selected mood
3. Retrieve mood genres
4. Filter candidate movies
5. Remove movies already rated by the user
6. Compute CF predictions
7. Compute CBF predictions
8. Combine predictions
9. Sort descending
10. Return Top-N
```

Keep this logic inside:

```text
src/hybrid.py
```

not inside the Streamlit UI file.

---

# 19. Cold-Start Handling

A practical recommendation system must handle cases where insufficient data exists.

## 19.1 Unknown User

If the user has no rating history, collaborative filtering cannot personalise effectively.

Possible fallback:

```text
Mood Match
+
Movie Average Rating
+
Popularity
```

Example fallback score:

```text
fallback_score = weighted_rating
```

Do not display a technical error.

Show:

```text
This user has limited rating history, so recommendations are based mainly on mood and movie popularity.
```

---

# 20. Sparse User Handling

Users with very few ratings may have weak collaborative signals.

Possible strategy:

```text
If number_of_user_ratings < threshold:
    reduce CF contribution
    increase CBF contribution
```

This is optional.

For the current assignment, a simpler fallback is acceptable as long as it is documented and the program does not fail.

---

# 21. No Result Handling

Some filtering combinations may produce too few results.

Example:

```text
if len(candidates) < requested_n:
    return available candidates
```

If zero results occur:

```text
No suitable movies were found for this selection.
Please try another mood.
```

Never allow the program to crash because the candidate list is empty.

---

# 22. Data Loading Strategy

Processed files currently include:

```text
movies_clean.csv
ratings_clean.csv
movie_lookup.csv
mood_genre_mapping.csv
```

The prototype should load processed data rather than repeat preprocessing every time it starts.

Recommended:

```python
@st.cache_data
def load_data():
    ...
```

For models or similarity matrices:

```python
@st.cache_resource
def load_model():
    ...
```

This makes the app faster.

---

# 23. Precompute Expensive Operations

Do not perform the following from zero every time a user presses Recommend:

- complete preprocessing;
- train/test split;
- full user-user similarity matrix;
- full movie similarity matrix;
- model evaluation;
- alpha tuning.

Instead:

```text
Offline Preparation
      ↓
Create Processed Data
      ↓
Create Models / Similarity Data
      ↓
Save
      ↓
Prototype Loads Saved Results
```

This reduces demo risk.

---

# 24. Performance

Recommendation results should ideally appear within a few seconds.

Target:

```text
Normal interaction:
< 3 seconds where practical
```

If a calculation takes longer:

- cache it;
- precompute it;
- optimise candidate selection;
- show a loading spinner.

Example:

```python
with st.spinner("Finding movies for you..."):
    ...
```

---

# 25. Validation and Error Handling

The assignment programming rubric values validation and logical handling.

Add validation for:

### Missing User

```text
Please select a user.
```

### Missing Mood

```text
Please select your current mood.
```

### Invalid User ID

```text
The selected user could not be found.
```

### Dataset Missing

```text
Required recommendation data could not be loaded.
```

### Model Missing

```text
Recommendation model is unavailable.
```

### Empty Candidate Set

```text
No suitable recommendations were found.
```

### Algorithm Failure

Catch exceptions and show a friendly message rather than a raw traceback.

During development, log the original error for debugging.

---

# 26. UI / UX Principles

The interface should be visually attractive but not overloaded.

## 26.1 Keep One Main Action

The main action should be:

```text
Recommend Movies
```

Do not show too many controls.

---

## 26.2 Use Consistent Layout

Recommended hierarchy:

```text
Page Title
Short Explanation
User Selection
Mood Selection
Recommendation Settings
Recommend Button
Results
```

---

## 26.3 Use Clear Labels

Prefer:

```text
Select your mood
```

instead of:

```text
Mood parameter
```

Prefer:

```text
Recommended for you
```

instead of:

```text
Prediction output
```

Technical language can appear on the evaluation page.

---

## 26.4 Use Whitespace

Avoid placing every element very close together.

Group related information inside:

- columns;
- containers;
- tabs;
- cards;
- expanders.

---

## 26.5 Avoid Excessive Colours

Use Streamlit's normal theme with a small number of consistent accents.

The page should look clean rather than colourful everywhere.

---

## 26.6 Keep Results Scannable

Users should be able to identify:

```text
Movie
Genre
Recommendation Strength
```

within a few seconds.

---

# 27. Movie Posters

Movie posters are optional.

They can improve visual appeal but should not become a dependency that may fail during demonstration.

Priority order:

```text
1. Working recommendation
2. Correct ranking
3. Reliable application
4. Clean layout
5. Poster images
```

If posters require live internet/API access, consider avoiding them for the final demo unless cached locally.

---

# 28. Page 2 — Algorithm Comparison

This page is useful for demonstrating each group member's contribution.

Possible layout:

```text
Select User
Select Mood
Select Algorithm

[Collaborative]
[Content-Based]
[Hybrid]
```

Display recommendations from each approach.

Optional:

```text
Compare All
```

which could display three columns.

Keep comparison manageable.

---

# 29. Algorithm Performance Table

Display the existing rating-prediction results:

| Method | RMSE | MAE |
|---|---:|---:|
| Collaborative Filtering | 0.9374 | 0.7317 |
| Content-Based Filtering | 0.9278 | 0.7186 |
| Hybrid | **0.8654** | **0.6745** |

Explain:

```text
Lower RMSE and MAE indicate better rating-prediction accuracy.
```

Do not claim that one algorithm is universally better beyond what the experiment supports.

---

# 30. Page 3 — Evaluation Results

Show evaluation separately from normal recommendations.

Recommended sections:

## Rating Prediction

- RMSE
- MAE

## Top-N Recommendation

- Precision@5
- Recall@5
- F1@5
- Precision@10
- Recall@10
- F1@10
- Precision@20
- Recall@20
- F1@20

Current hybrid results:

| K | Precision@K | Recall@K | F1@K |
|---:|---:|---:|---:|
| 5 | 0.1960 | 0.0326 | 0.0498 |
| 10 | 0.1390 | 0.0500 | 0.0613 |
| 20 | 0.0955 | 0.0712 | 0.0649 |

---

# 31. Evaluation Charts

Existing charts can be displayed rather than recalculated.

Possible files:

```text
output/evaluation_results.png
output/error_distribution.png
output/alpha_tuning.png
output/precision_recall_f1.png
output/mood_recommendations.png
```

For each chart, include a short explanation of what it demonstrates.

Do not show a chart without interpretation.

---

# 32. Avoid Mixing Evaluation and Live Recommendation Logic

The evaluation process and application recommendation process are related but different.

```text
Offline Evaluation
→ measures algorithm performance

Live Prototype
→ provides recommendations to a selected user
```

Do not rerun the entire experimental evaluation whenever a user requests recommendations.

---

# 33. Source-Code Structure

Recommended structure:

```text
mood-movie-recommender/
│
├── app.py
│
├── requirements.txt
├── README.md
│
├── data/
│   └── processed/
│       ├── movies_clean.csv
│       ├── ratings_clean.csv
│       ├── movie_lookup.csv
│       └── mood_genre_mapping.csv
│
├── src/
│   ├── collaborative.py
│   ├── content_based.py
│   ├── hybrid.py
│   ├── mood_filter.py
│   ├── data_loader.py
│   └── utils.py
│
├── models/
│   └── saved model / similarity files
│
├── output/
│   ├── evaluation_results.png
│   ├── error_distribution.png
│   ├── alpha_tuning.png
│   ├── precision_recall_f1.png
│   └── mood_recommendations.png
│
└── tests/
    └── optional test scripts
```

---

# 34. Responsibility of Each File

## `app.py`

Responsible for:

- page configuration;
- Streamlit components;
- user input;
- calling recommendation functions;
- displaying output.

Should not contain the full recommendation algorithm.

---

## `collaborative.py`

Responsible for:

- user-item representation;
- user similarity;
- neighbour selection;
- collaborative predictions;
- collaborative recommendations.

---

## `content_based.py`

Responsible for:

- content feature processing;
- TF-IDF;
- movie similarity;
- content-based predictions;
- content-based recommendations.

---

## `hybrid.py`

Responsible for:

- combining CF and CBF scores;
- alpha weighting;
- final ranking.

---

## `mood_filter.py`

Responsible for:

- reading mood mapping;
- identifying relevant genres;
- filtering movie candidates.

---

## `data_loader.py`

Responsible for:

- loading CSV files;
- checking required columns;
- handling missing files;
- cached loading if appropriate.

---

## `utils.py`

Responsible for reusable helper functions.

Do not place all unrelated code into this file.

---

# 35. Code Quality

Keep functions short and clearly named.

Prefer:

```python
get_mood_candidates()
predict_cf_rating()
predict_cbf_rating()
calculate_hybrid_score()
recommend_movies()
```

Avoid:

```python
run1()
calculate2()
final_function()
temp()
```

Add comments where logic is not obvious.

Do not add comments explaining very simple Python statements.

---

# 36. Configuration

Store reusable parameters in one place.

Example:

```python
TOP_K_NEIGHBOURS = 50
TFIDF_MAX_FEATURES = 500
HYBRID_ALPHA = 0.5
DEFAULT_N_RECOMMENDATIONS = 10
RELEVANCE_THRESHOLD = 3.5
```

This makes the system easier to explain and modify during Q&A.

---

# 37. Recommendation Quality Factors

A successful recommendation system should consider more than whether code executes.

Important factors include:

## 37.1 Relevance

Recommendations should match the user's likely preferences.

Measured partly using:

- Precision;
- Recall;
- F1;
- predicted-rating accuracy.

---

## 37.2 Personalisation

Different users should be capable of receiving different recommendations.

If every user receives the same list, the recommender has limited personalisation.

Test this before submission.

---

## 37.3 Context

This project adds mood as contextual information.

Different mood selections should produce meaningfully different candidate lists.

Test the same user with:

```text
Happy
vs
Excited
vs
Romantic
```

The recommendations should not always be identical.

---

## 37.4 Novelty

Avoid recommending only the most popular films every time.

Personalised scoring should allow less obvious but relevant items to appear where appropriate.

No dedicated novelty metric is required by the current assignment, so keep this as a design consideration rather than a claimed evaluation result.

---

## 37.5 Diversity

A Top-10 list containing ten nearly identical films may feel repetitive.

Possible simple improvement:

- avoid too many movies with exactly the same genre combination.

This is optional and should only be added if it does not interfere with the existing documented methodology.

---

## 37.6 Coverage

The mood mapping should provide enough movies for every mood.

Current preprocessing analysis shows all six moods have substantial movie coverage, so this requirement is already reasonably supported.

---

## 37.7 Accuracy

Predicted scores should be based on the implemented model rather than fabricated values.

Display real outputs from the algorithm.

---

## 37.8 Speed

Users should not wait unnecessarily long.

Optimise using:

- caching;
- preprocessing;
- saved similarity structures;
- smaller candidate sets where justified.

---

## 37.9 Reliability

The application should behave correctly for:

- valid users;
- sparse users;
- different moods;
- different Top-N values;
- missing data;
- empty results.

---

## 37.10 Transparency

The system should state:

- selected mood;
- selected algorithm;
- recommendation score where meaningful;
- basic reason for recommendation.

---

# 38. Important Consistency Requirement

The prototype should match the methodology described in the documentation.

If the documentation says:

```text
Mood
→ Genre Filter
→ Hybrid Recommendation
→ Top-N
```

the prototype should follow the same flow.

Avoid introducing a completely different recommendation method immediately before submission unless the documentation is also updated.

This is important for the system-implementation assessment.

---

# 39. Test Plan

Before submission, perform structured testing.

## 39.1 Functional Tests

Test:

- application opens;
- data loads;
- user selector works;
- each mood works;
- recommendation button works;
- requested Top-N count works;
- algorithm selection works;
- results are ranked;
- already-rated items are excluded.

---

## 39.2 Mood Tests

For the same user:

```text
Happy
Sad
Stressed
Excited
Romantic
Bored
```

Check that the returned movies correspond to the configured mood genres.

---

## 39.3 User Tests

Use:

```text
User with many ratings
User with medium number of ratings
User with few ratings
```

Check that the program remains stable.

---

## 39.4 Edge Tests

Test:

```text
Invalid user
Missing data
Empty recommendation candidates
Very small N
Large N
Unknown mood if manually passed
```

---

## 39.5 Comparison Tests

Verify:

```text
CF output
CBF output
Hybrid output
```

are generated from their respective functions.

---

# 40. Demo Preparation

Create at least three pre-tested demo scenarios.

Example:

```text
Demo 1
User: 123
Mood: Happy

Demo 2
User: 456
Mood: Excited

Demo 3
User: 789
Mood: Romantic
```

Record expected recommendation behaviour.

Do not depend on improvising the entire demonstration.

---

# 41. Lecturer Q&A Preparation

Each member should be able to explain:

```text
What algorithm did you implement?

Why was it selected?

What data does it use?

How is mood incorporated?

How are recommendations generated?

What does cosine similarity mean?

What is TF-IDF?

What is the hybrid formula?

Why is alpha 0.5?

What does RMSE measure?

What does MAE measure?

What do Precision and Recall measure?

What happens for a new user?

What are the limitations?
```

---

# 42. On-the-Spot Coding Preparation

Because the rubric includes on-the-spot coding, make the code easy to modify.

Be prepared for simple changes such as:

```text
Change Top-N from 10 to 5

Change hybrid alpha

Add another mood

Change a mood-to-genre mapping

Change the relevance threshold

Change UI text

Add a new displayed field
```

Do not create unnecessarily complicated code that makes small modifications difficult.

---

# 43. README Requirements

Create a simple `README.md`.

It should explain:

## Project

Mood-Based Movie Recommendation System

## Requirements

Example:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Main Features

- mood-based filtering;
- collaborative filtering;
- content-based filtering;
- hybrid recommendation;
- Top-N recommendation;
- evaluation results.

## Project Structure

Brief explanation of important folders.

---

# 44. `requirements.txt`

Keep dependencies minimal.

Example starting point:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
joblib
```

Generate final versions only after confirming the exact environment used.

---

# 45. Offline Demo Reliability

Assume the lecturer's machine or classroom internet may be unreliable.

The prototype should preferably work without internet access.

Therefore:

- store required data locally;
- store generated charts locally;
- store required model files locally;
- avoid mandatory external API calls;
- avoid remote images that must load successfully.

---

# 46. Data Size Consideration

The processed ratings dataset is much larger than the movie lookup data.

Consider whether all raw rating data must be loaded by the UI.

If a precomputed model or similarity structure can replace repeated heavy calculations, use it.

Do not remove data needed to reproduce the algorithm, because the lecturer requires source code.

Keep the preprocessing and training scripts/notebooks separately for reproducibility.

---

# 47. Reproducibility

The submitted project should allow another person to understand how outputs were produced.

Maintain:

```text
Preprocessing notebook
Evaluation notebook/script
Recommendation source files
Processed data
Required saved models
README
requirements.txt
```

If a model file is precomputed, document how it was generated.

---

# 48. Development Priorities

Use the following priority order.

## Priority 1 — Must Work

- data loading;
- mood selection;
- user selection;
- recommendation generation;
- Top-N display;
- correct algorithm connection.

## Priority 2 — Must Be Reliable

- validation;
- error handling;
- caching;
- testing;
- no crashes.

## Priority 3 — Must Be Explainable

- clean code;
- separated algorithm modules;
- clear labels;
- recommendation reasons;
- evaluation page.

## Priority 4 — Make It Look Good

- cards;
- emojis;
- spacing;
- polished headings;
- charts.

## Priority 5 — Optional Enhancements

- movie posters;
- diversity controls;
- advanced cold-start logic;
- extra charts.

Never complete Priority 5 while Priority 1 or Priority 2 is unfinished.

---

# 49. Suggested Development Stages

## Stage 1 — Create Basic Streamlit App

Build:

```text
Title
User selector
Mood selector
Recommend button
```

Use temporary results first if necessary to verify UI flow.

---

## Stage 2 — Connect Mood Filter

Verify:

```text
Mood
→ Correct Genres
→ Candidate Movies
```

---

## Stage 3 — Connect One Recommendation Algorithm

Start with the algorithm that is already easiest to call.

Verify real recommendation output.

---

## Stage 4 — Connect All Group Algorithms

Integrate:

```text
Collaborative
Content-Based
Hybrid
```

through consistent function interfaces.

Example:

```python
recommend(user_id, mood, n=10)
```

---

## Stage 5 — Improve Result Display

Add:

- ranking;
- title;
- genres;
- score;
- explanation.

---

## Stage 6 — Add Evaluation Page

Display:

- RMSE;
- MAE;
- Precision;
- Recall;
- F1;
- existing figures.

---

## Stage 7 — Add Validation

Handle all common failures.

---

## Stage 8 — Optimise Performance

Cache/precompute expensive components.

---

## Stage 9 — UI Polish

Improve:

- spacing;
- headings;
- containers;
- button labels;
- consistency.

---

## Stage 10 — Final Testing

Test all six moods and several users.

---

# 50. Minimum Successful Prototype

The minimum prototype worth submitting should have:

- [ ] Streamlit application opens without errors
- [ ] User can select a user profile
- [ ] User can select one of six moods
- [ ] User can request Top-N recommendations
- [ ] Mood filter is applied
- [ ] Recommendation algorithm produces real results
- [ ] Already-rated movies are excluded
- [ ] Results are ranked
- [ ] Movie title and genre are displayed
- [ ] Errors are handled
- [ ] Source code is organised
- [ ] Application can run locally
- [ ] `requirements.txt` exists
- [ ] `README.md` exists
- [ ] Evaluation results are available
- [ ] Each group member can explain their algorithm

---

# 51. Strong Prototype Checklist

For a stronger submission:

- [ ] Collaborative, content-based and hybrid models can be demonstrated
- [ ] Hybrid recommendation uses the documented alpha
- [ ] Recommendation explanations are shown
- [ ] Evaluation metrics are displayed clearly
- [ ] Evaluation charts are included
- [ ] Sparse-user fallback works
- [ ] The app works without internet
- [ ] Expensive operations are cached/precomputed
- [ ] Several prepared demo users are available
- [ ] Same user receives different recommendations for different moods
- [ ] Different users can receive different recommendations
- [ ] No raw Python tracebacks are shown to the user
- [ ] No major bugs occur during the demonstration
- [ ] Code is simple enough for on-the-spot modification

---

# 52. Final Recommendation

Build the prototype using:

```text
Python
+
Streamlit
+
Existing Processed Dataset
+
Existing Recommendation Algorithms
```

Focus on four things:

```text
Correct AI Logic
+
Simple User Experience
+
Reliable Execution
+
Clear Demonstration
```

A simple prototype that works correctly, produces real recommendations, handles errors and can be clearly explained is more suitable for this assignment than a visually complex application with unreliable AI functionality.

The final user experience should be:

```text
I choose who I am.
I choose how I feel.
I click one button.
I immediately understand the movies recommended to me and why.
```

That should remain the guiding principle throughout implementation.
