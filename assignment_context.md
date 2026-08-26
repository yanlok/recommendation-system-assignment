# Mood-Based Movie Recommendation System — AI Assignment Context

> This README consolidates the official **Artificial Intelligence Assignment Specifications** and the current **Assignment Documentation** for the project so that an AI coding assistant can understand the assignment requirements, project scope, report structure, and implementation constraints.

---

## 1. Assignment Overview

**Course:** Artificial Intelligence  
**Session:** 202605, Year 2026/27  
**Project Title:** Mood-Based Movie Recommendation System  
**Programme:** RSW2Y2S2  
**Tutorial Group:** 2  
**Tutor:** Ms Chai Foong Theng  

### Assignment Components

The assignment has two related components:

1. **Documentation — 40%**
2. **Prototype Development — 60%**

The project must identify a problem under one of the approved AI research areas, perform background/literature study, propose AI solution(s), implement the solution using Python or another relevant technology, test/evaluate it, and demonstrate the prototype.

### Submission

- **Documentation + Prototype Source Code**
- **Deadline:** 28 August 2026, Week 11, Friday, before 12:00 PM
- Submission is through Google Classroom.
- A prototype demonstration and Q&A are required during Weeks 12–14.

---

## 2. Selected Research Area

This project uses:

## Recommender System

The assignment requirements for a recommender-system project are:

- Identify a real-life scenario where a recommender system can suggest products, services, or content.
- Perform a background study on:
  - the selected problem/scenario;
  - the type(s) of recommender system to be implemented;
  - expected functionality and benefits.
- Each group member must provide a solution using a preferred recommender-system approach, for example:
  - collaborative filtering;
  - content-based filtering;
  - hybrid recommendation.
- Test and evaluate the recommender systems using suitable metrics, which may include:
  - Precision;
  - Recall;
  - F1 score;
  - Mean Squared Error (MSE);
  - Root Mean Squared Error (RMSE);
  - user satisfaction questionnaire.

The assignment expects different members to implement different approaches and compare their performance.

---

# 3. Project Problem

## 3.1 Background

Online streaming platforms contain very large collections of movies, which gives users more entertainment choices but also makes it more difficult to identify movies that match their interests and preferences.

Recommender systems help users discover relevant content by analyzing information such as:

- movie characteristics;
- user interactions;
- past ratings or preferences.

Common recommender-system approaches include:

- content-based filtering;
- collaborative filtering;
- hybrid recommendation.

However, movie preference may also change depending on a user's current emotional state.

Examples:

- a stressed user may prefer relaxing or comedy movies;
- an excited user may prefer action or adventure movies;
- a user who normally likes action movies may prefer a light-hearted movie when feeling stressed.

The proposed system therefore introduces **mood as an additional recommendation context**.

---

## 3.2 Problem Statement

Conventional recommender systems commonly rely on:

- previous ratings;
- user interaction history;
- movie characteristics.

These factors may not fully represent a user's **immediate viewing preference**.

A user's emotional state may influence what they want to watch at a particular time. Therefore, recommendations based only on historical preference may sometimes be unsuitable for the user's current mood.

The proposed system aims to use a **user-selected mood** as an additional factor so that movie recommendations are more context-aware, personalized, interactive, and relevant to the user's current emotional state.

---

# 4. Project Objective

The current documentation has not yet finalized Section 1.3 Objectives/Aims.

The final objectives should remain aligned with the project problem and assignment requirements.

Possible project-level objectives that still require team confirmation include:

1. Develop a movie recommendation system that incorporates the user's selected mood into the recommendation process.
2. Implement different recommender-system algorithms for the same mood-based movie recommendation problem.
3. Evaluate and compare the recommendation approaches using appropriate quantitative metrics.
4. Build a working prototype that allows users to select a mood and receive suitable movie recommendations.

> **Important:** Treat these as working objectives until the team finalizes the official wording in the report.

---

# 5. Current Team Information

| No. | Student | Student ID | Module / Algorithm in Charge |
|---|---|---:|---|
| 1 | Tay Ernest | 2501307 | Collaborative Filtering |
| 2 | Chay Qian | 2501538 | Not yet specified in the current document |
| 3 | Wong Jin Yu | 2501126 | Content-Based Filtering |

The official assignment requires each group member to develop and present their own work.

---

# 6. Core Project Design

The project should stay focused on **one common problem**:

> Recommend movies according to the user's current mood.

Different algorithms should solve this same problem rather than creating unrelated systems.

A conceptual structure is:

```text
User selects mood
        ↓
Mood information is converted into usable recommendation context
        ↓
Different recommender algorithms process the same overall problem
        ↓
Generate movie recommendations
        ↓
Evaluate and compare algorithms
```

Possible algorithm assignments include:

```text
Member 1 → Collaborative Filtering
Member 2 → Content-Based Filtering
Member 3 → Hybrid / another recommender approach
```

The exact algorithm assignment for each member must be confirmed by the group.

---

# 7. Dataset

## 7.1 Current Dataset Choice

The team is currently preparing to use:

**TMDB 5000 Movie Dataset with Ratings**  
Source selected by the team: Kaggle

Dataset page:

`https://www.kaggle.com/datasets/aayushsoni4/tmdb-5000-movie-dataset-with-ratings`

The final report must properly acknowledge/cite the dataset source.

---

## 7.2 Dataset Requirements

Before model implementation, inspect the actual downloaded files and confirm:

- file names;
- number of rows and columns;
- movie identifier fields;
- user identifier fields;
- rating fields;
- title;
- genres;
- keywords;
- overview;
- missing values;
- duplicate records;
- rating scale;
- number of unique users;
- number of unique movies;
- how the movie table connects to the ratings table.

Do **not** assume field names or relationships until they are verified from the downloaded dataset.

---

## 7.3 Recommended Project Data Structure

```text
project/
│
├── data/
│   ├── raw/
│   │   └── original downloaded dataset files
│   │
│   └── processed/
│       └── cleaned / transformed data
│
├── notebooks/
│   └── dataset exploration and experiments
│
├── src/
│   └── implementation code
│
├── README.md
└── requirements.txt
```

Keep original downloaded data unchanged inside `data/raw/`.

Any preprocessing output should be saved separately inside `data/processed/`.

---

# 8. Dataset Exploration Checklist

The first development stage should inspect the dataset before selecting final implementation details.

Example Python workflow:

```python
import pandas as pd

movies = pd.read_csv("path/to/movie_file.csv")
ratings = pd.read_csv("path/to/ratings_file.csv")
```

Inspect:

```python
movies.head()
ratings.head()
```

```python
movies.shape
ratings.shape
```

```python
movies.columns.tolist()
ratings.columns.tolist()
```

```python
movies.info()
ratings.info()
```

Missing values:

```python
movies.isnull().sum()
ratings.isnull().sum()
```

Duplicates:

```python
movies.duplicated().sum()
ratings.duplicated().sum()
```

Ratings:

```python
ratings["rating"].describe()
```

Users and ratings:

```python
ratings["userId"].nunique()
len(ratings)
```

The exact code must be adjusted to the real column names in the downloaded dataset.

---

# 9. Mood Representation

The selected dataset may contain movie metadata and ratings but may not contain an explicit user mood field.

Therefore, the project must define how mood is represented.

Possible user input:

```text
Happy
Sad
Stressed
Excited
Romantic
Bored
```

A possible design is to map moods to suitable movie genres or other metadata.

Example only:

```text
Happy     → Comedy / Animation / Family
Excited   → Action / Adventure / Thriller
Romantic  → Romance / Drama
```

> These mappings are **not provided by the assignment specification** and should not be treated as established facts. The team must define and justify the final mapping, preferably using relevant literature or a clearly explained rule-based design.

All algorithms should use the same agreed mood framework so that comparison remains fair.

---

# 10. Methodology Structure Required by the Report

The official documentation template requires:

## 3.1 System Flowchart / Activity Diagram

Draw and describe a simple diagram illustrating:

- system design;
- system flow;
- data flow.

This section may be completed after the dataset and implementation workflow are clearer, but it must be included in the final submission.

---

## 3.2 Description and Analysis of Dataset

Explain:

- dataset source;
- dataset structure;
- characteristics;
- important properties;
- preprocessing;
- feature extraction;
- preliminary data analysis.

---

## 3.3 Algorithm Selection & Description

Describe:

- which algorithm(s) are selected;
- why they are appropriate;
- how they are used in this project.

Each team member should explain their own recommender-system approach.

---

## 3.4 Evaluation Metrics

Explain the quantitative measures used to assess and compare the recommender systems.

Potential metrics allowed by the assignment include:

- Precision;
- Recall;
- F1 Score;
- MSE;
- RMSE;
- user satisfaction.

The final metrics should match the type of recommendation output produced.

---

# 11. Suggested Development Workflow

```text
1. Download dataset
        ↓
2. Inspect all files and columns
        ↓
3. Understand relationships between movie and rating data
        ↓
4. Analyze missing values / duplicates / data quality
        ↓
5. Clean and preprocess dataset
        ↓
6. Extract useful movie features
        ↓
7. Define and justify mood representation
        ↓
8. Prepare common data for all algorithms
        ↓
9. Implement each member's recommender algorithm
        ↓
10. Integrate mood into each algorithm
        ↓
11. Generate Top-N movie recommendations
        ↓
12. Evaluate algorithms
        ↓
13. Compare results
        ↓
14. Build final interface / prototype
        ↓
15. Complete system flowchart
        ↓
16. Finish methodology, results and discussion
```

---

# 12. Collaborative Filtering Module

The current documentation assigns **Collaborative Filtering** to Tay Ernest.

Possible collaborative-filtering approaches include:

### User-Based Collaborative Filtering

```text
Find users with similar rating behaviour
        ↓
Identify movies liked by similar users
        ↓
Predict / rank unseen movies
        ↓
Apply mood context
        ↓
Recommend Top-N movies
```

### Item-Based Collaborative Filtering

```text
Find movies with similar rating patterns
        ↓
Use movies already liked by the user
        ↓
Predict / rank related movies
        ↓
Apply mood context
        ↓
Recommend Top-N movies
```

### Matrix Factorization / SVD

```text
User–Movie Rating Matrix
        ↓
Matrix Factorization / SVD
        ↓
Learn user and movie latent factors
        ↓
Predict unseen ratings
        ↓
Apply mood context
        ↓
Recommend Top-N movies
```

The final collaborative-filtering approach should be selected after inspecting the dataset and confirming that the ratings structure supports it.

---

# 13. Evaluation

The group must compare different recommender approaches.

Depending on implementation, suitable metrics may include:

## Rating Prediction

If algorithms predict numerical ratings:

- MSE
- RMSE

## Top-N Recommendation

If algorithms return ranked movie recommendations:

- Precision@K
- Recall@K
- F1@K

## User-Oriented Evaluation

If required:

- questionnaire;
- user satisfaction rating;
- perceived relevance.

All algorithms should be evaluated under reasonably consistent conditions so that comparison is meaningful.

---

# 14. Results & Discussion Requirements

The report requires:

## 4.1 Results

Present outcomes using suitable:

- tables;
- figures;
- charts.

## 4.2 Discussion / Interpretation

Discuss:

- what the results mean;
- which algorithm performed better;
- why performance differs;
- how mood integration affected recommendations;
- trade-offs between approaches;
- whether objectives were achieved.

---

# 15. Conclusion Requirements

## 5.1 Achievements

Explain:

- what the project successfully implemented;
- whether the objectives were fulfilled.

## 5.2 Limitations and Future Work

Discuss:

- weaknesses of the current dataset;
- limitations of mood representation;
- cold-start issues;
- sparse ratings;
- limited user testing;
- possible future algorithms;
- richer emotional/context information;
- improved UI or real-time integrations.

Only include limitations that genuinely apply to the final implementation.

---

# 16. Documentation Marking Rubric

The documentation is worth **40%**.

The rubric evaluates:

### Introduction

Strong submissions should include:

- comprehensive background;
- clear problem statement;
- justified research gap;
- aligned objectives;
- significance.

### Related Work

Strong submissions should:

- critically analyze previous studies;
- compare methods;
- identify limitations;
- clearly justify the research gap.

### Methodology

Strong submissions should clearly justify:

- system flow;
- dataset;
- algorithms;
- evaluation metrics.

### Results & Discussion

Strong submissions should:

- present comprehensive results;
- clearly interpret outcomes;
- connect discussion to objectives.

### Conclusion, References & Sources

Strong submissions should:

- state achievements;
- acknowledge limitations;
- propose thoughtful future improvements;
- use complete APA-style references;
- acknowledge datasets and development tools.

---

# 17. Prototype Marking Rubric

The prototype is worth **60%**.

Main criteria:

## User Interface / Output — 10%

Strong work should:

- generate all necessary outputs;
- provide accurate recommendations/results;
- use a well-organized layout.

## Programming — 20%

Strong work should demonstrate:

- correct logical flow;
- suitable algorithmic complexity;
- good programming skills;
- exception handling;
- validations;
- implementation of important rules.

## Degree of Completion — 10%

Strong work should:

- contain all required features;
- run successfully;
- avoid bugs during demonstration.

## System Implementation — 10%

The final system should conform to the proposed system design.

## Presentation and On-the-Spot Coding — 10%

Each student must understand and be able to explain their own code and implementation decisions.

---

# 18. Related Work

The current documentation still needs:

## 2.1 Review of Previous Studies

The final section should:

- summarize relevant previous research;
- explain methods used;
- discuss key findings;
- compare different recommender-system approaches;
- review emotion-aware / mood-aware recommendation work where relevant.

## 2.2 Research Gap and Justification

The final section should:

- identify limitations in previous studies;
- explain unresolved issues;
- justify why the proposed mood-based system is worth studying.

Do not invent studies or citations.

Every study included in the final report should be checked against its original source and referenced using APA style.

---

# 19. Existing Citations in the Current Draft

The current document already cites:

- Zhang et al. (2023)
- de Campos et al. (2023)
- Winoto & Tang (2010)
- Polignano et al. (2021)

Before final submission:

- verify each reference against its original publication;
- ensure all bibliographic details are correct;
- include them in the References section using APA format;
- do not rely on AI-generated citation details without verification.

---

# 20. Academic Integrity

The assignment requires original work.

Students must:

- work only with their own team members;
- avoid copying another group's idea, report, or code;
- maintain evidence of individual contributions;
- comply with TARUMT plagiarism requirements.

Each student must complete the required plagiarism declaration.

---

# 21. AI Usage Requirement

AI tools are allowed as collaborative learning tools for activities such as:

- brainstorming;
- coding;
- refining written work;
- research support;
- editing.

However, **all AI usage must be disclosed**.

Each student must complete the AI Usage Disclosure Form and specify:

- AI tool(s) used;
- how they were used;
- prompts / type of assistance;
- how generated information was verified.

Students remain responsible for:

- factual accuracy;
- code correctness;
- citation accuracy;
- reasoning;
- final interpretation.

---

# 22. Instructions for an AI Coding Assistant

When helping with this project, follow these rules:

1. Keep the project focused on **Mood-Based Movie Recommendation**.
2. Treat all algorithms as alternative solutions to the same recommendation problem.
3. Do not redesign the project into an unrelated movie recommender.
4. Do not assume dataset columns without inspecting the actual downloaded files.
5. Keep raw data unchanged.
6. Explain preprocessing decisions.
7. Do not invent research papers, dataset properties, results, or citations.
8. Clearly separate:
   - facts from the assignment;
   - facts observed in the dataset;
   - design decisions proposed by the team.
9. Ensure mood handling is applied consistently across algorithms where comparison requires it.
10. Prefer code that is understandable enough for students to explain during the demo and Q&A.
11. Include validation and error handling where appropriate.
12. Record evaluation results so they can be used in the report.
13. Keep the final implementation aligned with the documented methodology.
14. When suggesting a new algorithm or design change, explain why it fits the assignment.
15. Do not claim that a method has better performance until actual evaluation results exist.

---

# 23. Current Project Status

Completed / started:

- [x] Project title selected
- [x] Recommender System research area selected
- [x] Background drafted
- [x] Problem Statement drafted
- [x] Collaborative Filtering assigned to one member
- [x] Candidate Kaggle dataset selected
- [ ] Objectives finalized
- [ ] Significance / contribution written
- [ ] Related Work completed
- [ ] Research gap finalized
- [ ] Dataset downloaded and inspected
- [ ] Dataset preprocessing completed
- [ ] Mood representation finalized
- [ ] Algorithms finalized for all members
- [ ] System flowchart completed
- [ ] Evaluation metrics finalized
- [ ] Prototype implemented
- [ ] Results collected
- [ ] Algorithms compared
- [ ] Discussion written
- [ ] Conclusion written
- [ ] References completed and verified
- [ ] AI disclosure completed
- [ ] Final demo prepared

---

# 24. Immediate Next Task

The next development task is:

> **Inspect the downloaded TMDB dataset before implementing any recommendation algorithm.**

Required outputs to record:

```python
movies.head()
ratings.head()

movies.shape
ratings.shape

movies.columns.tolist()
ratings.columns.tolist()

movies.info()
ratings.info()

movies.isnull().sum()
ratings.isnull().sum()
```

Also verify:

- the movie identifier;
- the rating identifier;
- the user identifier;
- how ratings connect to movies;
- genre format;
- rating scale;
- number of unique users;
- number of unique movies.

Only after this inspection should preprocessing and final algorithm design continue.

---

# 25. Source Documents

This README was consolidated from:

1. **Artificial Intelligence Assignment Specifications — Session 202605**
2. **Artificial Intelligence Assignment Documentation — Mood-Based Movie Recommendation System**

The official assignment documents remain the authoritative source if any conflict arises between this README and the original files.
