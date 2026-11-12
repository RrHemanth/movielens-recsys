# MovieLens 25M Recommender: from raw data to a monitored service

I built this project to learn how a recommender system works end to end. It is not just a model in a notebook: it covers verified data, honest time-based evaluation, several models that each had to beat the previous one, statistics to check that the improvements are real, an API in Docker, drift monitoring, CI, and a retraining workflow.

Everything here is **offline**: I evaluate on historical MovieLens data, held out by time. There are no live users and no A/B tests. Where something is simulated, I say so.

---

## Contents

1. [The short version](#1-the-short-version)
2. [The problem](#2-the-problem)
3. [The data and what I learned from it](#3-the-data-and-what-i-learned-from-it)
4. [How I evaluate (and why it's done this way)](#4-how-i-evaluate-and-why-its-done-this-way)
5. [The models, step by step](#5-the-models-step-by-step)
6. [Are the improvements real? Statistics](#6-are-the-improvements-real-statistics)
7. [Final test on 2019](#7-final-test-on-2019)
8. [Serving: FastAPI + Docker](#8-serving-fastapi--docker)
9. [Monitoring: drift and retraining](#9-monitoring-drift-and-retraining)
10. [Dashboard](#10-dashboard)
11. [Engineering: reproducibility, tests, CI](#11-engineering-reproducibility-tests-ci)
12. [How to run it yourself](#12-how-to-run-it-yourself)
13. [Repository layout](#13-repository-layout)
14. [Limitations and honest caveats](#14-limitations-and-honest-caveats)
15. [What I would do next](#15-what-i-would-do-next)
16. [Acknowledgements](#16-acknowledgements)

---

## 1. The short version

**Goal:** given a user's past movie ratings, recommend 10 movies they will go on to like.

**System:**
- **Returning users:** a two-stage recommender. First, a shortlist of about 150 movies per user from **ALS matrix factorization** (PySpark) and **recent popularity**. Then a **LightGBM re-ranker** picks the final 10.
- **New users:** after they rate a few movies, I place them into the ALS "taste space" by **averaging the vectors of the movies they liked** (fold-in, no retraining).
- **Unknown users:** recent popularity.

**Final results on a held-out year (2019), NDCG@10:**

| Users | Baseline (recent popularity) | My system | Improvement |
|---|---|---|---|
| Returning (4,655 users) | 0.0897 | **0.1338** (re-ranker) | **+49%**, and +13% over ALS alone |
| New, after 5 ratings (5,697 users) | 0.3708 | **0.3970** (fold-in) | **+7%**, with ~100× more catalog coverage |

All of these differences are statistically significant (paired bootstrap and sign-flip test, p < 0.001).

**Ops:** MLflow experiment tracking, FastAPI service in Docker (192 MB image), PSI/KS drift monitoring, GitHub Actions for tests on every push and a drift-triggered retraining workflow, and a Streamlit dashboard. 79 automated tests.

---

## 2. The problem

Most data science problems predict one label per row (will this customer churn?). A recommender is different: for each user, it has to **rank tens of thousands of movies** and put the right ones in the top 10. So this is a **ranking problem**, and that framing drives most of the choices below.

A useful mental picture is a giant grid:
- one **row per user** (162,541 users)
- one **column per movie** (about 59,000 movies that have ratings)
- each **cell** holds the user's rating, or is **empty** if they never rated that movie

The recommender's job is to guess which empty cells a user would fill with a high rating, and recommend those movies.

---

## 3. The data and what I learned from it

### The dataset

[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) from GroupLens (University of Minnesota): real ratings from the MovieLens website.

| | |
|---|---|
| Ratings | 25,000,095 (0.5 to 5 stars, in half-star steps) |
| Users | 162,541 |
| Movies | 62,423 listed, 59,047 with at least one rating |
| Time span | 1995-01-09 to 2019-11-21 |

I mainly use two tables: `ratings` (user, movie, stars, timestamp) and `movies` (title, genres). The `genome` tag-relevance tables are kept for later; I skipped user `tags` because they are sparse and messy free text.

### Getting the data right first

- **Downloaded with code**, so anyone can rebuild it with one command.
- **Verified** against the official MD5 checksum (`6b51fb2759a8657d3bfcbfc42b592ada`), and my row counts match the dataset's README exactly.
- **Converted CSV to Parquet** with DuckDB, choosing every column type explicitly instead of letting a tool guess. For example, IMDb IDs keep their leading zeros, and there is a test for that. The data shrank from **1.1 GB to 150 MB** (ratings alone: 647 MB → 117 MB), and the conversion takes about 1.5 seconds.

### What the data looks like (from SQL queries in `sql/eda/`)

| Finding | Number | What it meant for the project |
|---|---|---|
| The grid is almost empty | only **0.26%** of user–movie pairs are rated | use matrix factorization, which is built for sparse data |
| Popularity is very concentrated | top **1%** of movies get **47.6%** of all ratings; top 10% get **94%** | a popularity baseline will be strong, so every model must beat it; also track coverage and diversity |
| The long tail | **58.8%** of movies have fewer than 10 ratings; the median movie has 6 | collaborative filtering will struggle with most movies |
| Some movies have no ratings | **3,376** movies | item cold start |
| Users vary a lot | median **71** ratings, mean **153.8**, max **32,202** | compute metrics per user, then average, so heavy users don't dominate |
| Every user has at least 20 ratings | a dataset filter | no natural cold-start users in the raw data (this changes once you split by time, see below) |
| Ratings lean positive | **4.0** is the most common rating; about **50%** are ≥ 4.0 | define "liked" as rating ≥ 4.0 |

---

## 4. How I evaluate (and why it's done this way)

### Split by time, not randomly

A random split lets the model learn from the future: it could train on a user's 2019 ratings and be "tested" on their 2015 ratings. That is data leakage, and the scores would look much better than anything achievable in real use. So I split by time, like a forecasting problem:

| Part | Period | Ratings | Users | % rated ≥ 4 |
|---|---|---|---|---|
| Train | before 2018 | 22,488,700 | 150,543 | 49.8% |
| Validation | 2018 | 1,310,761 | 12,001 | 49.6% |
| Test | 2019 | 1,200,634 | 10,601 | 49.6% |

I picked the cutoffs from the data: 90% of ratings happened before early January 2018, and 95% before late December 2018, so calendar years were a natural choice (about 90/5/5). The share of 4+ ratings is stable across all three periods, so "liked" means the same thing in each.

**All model choices were made on validation (2018). The test year (2019) was used once, at the very end.**

### A surprise: most evaluation users are new

| | Users | Seen before the period | New |
|---|---|---|---|
| Validation (2018) | 12,001 | 5,701 (47.5%) | 6,300 (52.5%) |
| Test (2019) | 10,601 | 4,903 (46.3%) | 5,698 (53.7%) |

More than half the users in each period had never rated anything before. Matrix factorization can't score users it never saw, so cold start is the main case here, not an edge case. **I report returning and new users separately everywhere.** One blended number would hide very different behavior.

### What counts as a good recommendation

For each user, the model recommends 10 movies they have not rated yet. A recommended movie is a **hit** if the user rated it **4 stars or higher** during the evaluation period.

| Metric | Measured on | Question it answers |
|---|---|---|
| **NDCG@10** (main metric) | each user, then averaged | Did we find the movies they liked, and put them near the top? A hit at position 1 is worth more than a hit at position 10. Between 0 and 1. |
| **Recall@10** | each user, then averaged | Of the movies they liked, how many did we find? |
| **Coverage** | all users together | What fraction of the catalog is ever recommended to anyone? |
| **Diversity** | each list, then averaged | How different are the genres within one list? (1 − Jaccard similarity of genres, averaged over pairs) |

I wrote all four metrics myself and tested them against small examples I computed by hand (`tests/test_metrics.py`).

---

## 5. The models, step by step

Each model had to beat the previous one on **returning users** in validation (2018). Every run is logged in MLflow.

### 5.1 Popularity baseline

Recommend the most-liked movies (ratings ≥ 4), skipping ones the user has already rated. The interesting question was "popular *when*?":

| Popularity window | NDCG@10 (returning) | NDCG@10 (new) |
|---|---|---|
| All-time | 0.0597 | 0.3927 |
| Last 365 days | 0.0931 | 0.4480 |
| **Last 90 days** | **0.1131** | 0.4452 |

Recent popularity almost doubles the all-time score, because what people watched recently is the best guess for what they'll watch next.

**Why do new users score so much higher?** I checked with a query: in 2018, new users liked about **82 movies each** versus about **23** for returning users. New users are catching up on famous movies they've seen over the years, which is exactly what a popularity model recommends. Returning users already rated those years ago, so the model has to skip them and dig deeper. This is why the honest bar for personalized models is **0.1131 on returning users**.

### 5.2 ALS matrix factorization (PySpark)

ALS gives every user and every movie a vector of 64 learned "taste numbers"; a user's score for a movie is the dot product of the two vectors. I used the **implicit** version: ratings ≥ 4 become "liked = 1", and everything else (including the empty cells) is a weak 0. Predicting exact star ratings tends to push obscure movies with one enthusiastic rating to the top, which hurts a top-10 list.

Trained on all history, ALS **lost** to popularity (0.1093 vs 0.1131). Since recency had mattered so much for popularity, I tried learning only from recent years:

| ALS learns from | NDCG@10 (returning) |
|---|---|
| All years | 0.1093 |
| 2010 onward | 0.1150 |
| 2015 onward | 0.1206 |
| 2017 only | 0.1273 |

Then I tuned alpha (how much a "like" outweighs an empty cell) and regularization over a 3×3 grid, extending the grid where the best value sat on its edge:

| Setting (2017 window, rank 64) | NDCG@10 | Coverage |
|---|---|---|
| alpha 1, reg 0.1 | 0.1315 | 0.0309 |
| alpha 10, reg 1.0 | 0.1300 | 0.0507 |

These two turned out to be statistically tied (see section 6), so I used **alpha 10, reg 1.0**, which recommends 64% more distinct movies.

A caveat about Spark: 22 million ratings fit comfortably in my laptop's memory, so Spark wasn't strictly necessary. I ran it in local mode to learn Spark's ALS and so the same code could run on a cluster.

### 5.3 Similar movies with FAISS

ALS movie vectors can also find similar movies. Without ever seeing genres or directors, it found that *Inception* is close to *Interstellar*, *The Dark Knight* and *The Prestige* (all Christopher Nolan films), and *Toy Story* is close to other Pixar films.

I benchmarked FAISS exact search against an approximate IVF index on the 20,450 movie vectors:

| Search | Time per movie | Recall of the true 10 neighbors |
|---|---|---|
| Exact | ~0.007 ms | 1.000 |
| Approximate, 1 of 128 clusters | ~0.007 ms | 0.732 |
| Approximate, 16 of 128 clusters | ~0.008 ms | 0.969 |
| Approximate, 64 of 128 clusters | ~0.011 ms | 0.990 |

At this size, the approximate index gave **no speed-up**, because exact search is already one fast matrix multiplication. Approximate search pays off at millions of items, so I kept exact search. When FAISS and NumPy disagreed on 23% of top-10 lists, I checked every case: all were ties, because about **34% of movies have a near-identical "twin" vector** (most likely rarely-liked movies the model can't tell apart).

### 5.4 Two-stage: candidates + LightGBM re-ranker

**Stage 1, candidates:** for each user, merge their ALS top 100 with the 90-day popular top 100 (about 150 unique movies). The shortlist contains about **32%** of the movies each user goes on to like, versus 20% for ALS alone and 24% for popularity alone. The re-ranker can only reorder what's on the shortlist, so this is its ceiling.

**Stage 2, re-ranking:** for each (user, candidate) pair, compute 8 features:

| Feature | Meaning |
|---|---|
| `als_score`, `als_rank` | taste match from ALS |
| `genre_match` | how well the movie's genres match the genres of the user's past likes |
| `likes_90d`, `likes_365d` | how popular the movie is right now / this year |
| `movie_year` | release year (from the title) |
| `user_likes_365d` | how active the user is |
| `in_both_lists` | was it suggested by both ALS and popularity? |

**Avoiding leakage was the trickiest part.** The re-ranker needs examples with known answers, but the features must never be built from the answers. So I built its training data **one year earlier**: features computed only from data up to the end of 2016, and labels from what users liked in 2017. It is then applied to features as of the end of 2017 and graded on 2018. Every feature is recomputed "as of" the pretend date (popularity from the last 90 days before it, ALS trained on the year before it), and the code asserts that no rating on or after that date is used.

| Validation (2018), returning users | NDCG@10 | Coverage |
|---|---|---|
| ALS (shortlist setting) | 0.1300 | 0.0507 |
| **Re-ranker, binary classification** | **0.1428** | 0.0321 |
| Re-ranker, lambdarank (learning to rank) | 0.1404 | 0.0280 |

The re-ranker beats ALS by about 10%, but **coverage drops**: it leans toward currently popular titles. Binary and lambdarank were statistically tied, so I kept the simpler binary version. The most important features were the ALS score (binary) and release year, recent popularity and ALS rank (lambdarank).

### 5.5 Cold start for new users

To test cold start honestly, I **simulated onboarding**: each new user's first 5 ratings in the period (in time order) count as known, and every method is graded on that user's later ratings. Popularity is re-scored under the exact same setup.

| New users, validation (2018) | NDCG@10 | Coverage |
|---|---|---|
| Popularity | 0.3533 | 0.0003 |
| **Fold-in** (average the ALS vectors of liked onboarding movies) | **0.3808** | 0.0348 |
| Blend of fold-in and popularity (rank average) | 0.3681 | 0.0169 |

Five ratings are enough to personalize: fold-in beats popularity by about 8%, with no retraining. I expected the blend to win; it didn't. A likely reason is that the averaged vector already favors popular movies, so mixing in popularity again over-weights it.

---

## 6. Are the improvements real? Statistics

Every score is an average over a few thousand users, so I checked whether differences could be luck:
- **95% bootstrap confidence intervals:** resample users with replacement 10,000 times and take the middle 95% of the averages.
- **Paired comparisons:** compare two models user by user (difference = model B − model A for the same user), then bootstrap the average difference. This removes the huge variation in how hard individual users are.
- **Sign-flip test** for a p-value: if two models were equally good, each user's difference could just as well have the opposite sign.

Validation (2018), returning users:

| Comparison | Gain in NDCG@10 | 95% interval | p-value | Verdict |
|---|---|---|---|---|
| ALS (best) vs popularity | +0.0184 | +0.0139 to +0.0227 | < 0.001 | real |
| ALS: two shortlisted settings | +0.0015 | −0.0012 to +0.0041 | 0.27 | tie |
| Re-ranker vs its ALS input | +0.0128 | +0.0088 to +0.0168 | < 0.001 | real |
| Re-ranker vs best ALS | +0.0113 | +0.0070 to +0.0157 | < 0.001 | real |
| Binary vs lambdarank | +0.0024 | −0.0002 to +0.0052 | 0.08 | tie |
| Fold-in vs popularity (new users) | +0.0275 | +0.0236 to +0.0313 | < 0.001 | real |

The significant results survive a strict correction for running several comparisons. Even a real average gain is not uniform: the re-ranker helped about 32% of returning users and hurt about 28% (the rest were unchanged).

---

## 7. Final test on 2019

All choices were frozen on validation. I then rebuilt every component "as of" January 1, 2019 (popularity from Oct–Dec 2018, ALS trained on 2018 likes, the re-ranker retrained on its most recent labeled year), and evaluated **once** on 2019.

**Returning users (4,655):**

| System | NDCG@10 [95% CI] | Recall@10 | Coverage |
|---|---|---|---|
| Popularity (90 days) | 0.0897 [0.0850, 0.0945] | 0.0473 | 0.0046 |
| ALS | 0.1187 [0.1134, 0.1239] | 0.0655 | 0.0450 |
| **Re-ranker** | **0.1338 [0.1285, 0.1391]** | **0.0776** | 0.0295 |

**New users after 5 onboarding ratings (5,697):**

| System | NDCG@10 [95% CI] | Recall@10 | Coverage |
|---|---|---|---|
| Popularity | 0.3708 [0.3636, 0.3781] | 0.0719 | 0.0003 |
| **Fold-in** | **0.3970 [0.3897, 0.4041]** | **0.0795** | **0.0305** |

**Paired tests on 2019** (all p < 0.001): re-ranker vs popularity **+0.0440**; re-ranker vs ALS **+0.0151**; ALS vs popularity +0.0289; fold-in vs popularity **+0.0262**.

I expected the gains to shrink on the test year. They didn't: the re-ranker's gain over ALS was slightly larger than on validation, and the gap to popularity grew, mostly because the popularity baseline itself got worse in 2019. The new-user gain matched validation almost exactly.

To reproduce: `uv run python scripts/final_test.py` (it's also logged in MLflow as "FINAL TEST 2019").

---

## 8. Serving: FastAPI + Docker

**Batch + online design.** Running the whole pipeline (Spark, features, LightGBM) on every request would be far too slow. So:
- A **batch job** (`scripts/build_serving_artifacts.py`) precomputes the top 10 for the **12,001 users active in the last year** and saves small files (~15 MB) plus a `manifest.json` recording the snapshot date, settings and git commit.
- The **API** loads those files once and answers from memory.
- **New users** are handled **live**: fold-in is just a bit of NumPy math.

| Endpoint | What it does |
|---|---|
| `GET /recommendations/{user_id}` | returning user → precomputed re-ranker list; unknown user → popularity |
| `POST /recommendations/new-user` | body `{"liked_movie_ids": [...]}` → live fold-in |
| `GET /health` | status, snapshot date, number of precomputed users, git commit |

Every response includes a `source` field (`re-ranker`, `fold-in` or `popularity`) so you can see which model produced it. Inputs are validated with Pydantic, and invalid requests get a 422.

**Docker:** the image contains only what serving needs (no Spark, Java, LightGBM or MLflow). I split the dependencies into `core`, `train`, `dashboard` and `dev` groups for this. The serving snapshot is baked in, so each image is a versioned model; it runs as a non-root user and has a health check. Image size: **192 MB** compressed (788 MB on disk).

---

## 9. Monitoring: drift and retraining

**Drift** means the incoming data no longer looks like what the model learned from. I replayed history month by month (Jan 2018 to Nov 2019) for a model built on 2017 data and **never retrained**, comparing each month with 2017 using the **Population Stability Index (PSI)** and the KS statistic:

| Signal | What happened |
|---|---|
| Ratings on movies the model has **no vector** for | rose from **1.6%** (reference) to **12.6%**; PSI crossed **0.1 ("watch") in December 2018**, peaking at 0.24 |
| Ratings per user, movie age, rating values | stable (PSI < 0.1 throughout, except one partial month) |
| Share of ratings from users the model never saw | 66% to 85% |

New films keep coming out, and a frozen model can't score any of them: that's the clearest staleness signal. I don't rely on KS p-values, because with hundreds of thousands of ratings they flag even trivial changes; PSI thresholds drive decisions.

A lesson here: the common "PSI > 0.25 = drift" rule would never have fired, even with 12% of ratings on unknown movies. I set the retraining trigger at **PSI ≥ 0.1 on any signal**, which fires about 12 months after training.

**Retraining workflow** (`.github/workflows/retrain.yml`): monthly, or on demand, GitHub Actions rebuilds the data from the official source, runs the drift check, and only if triggered, rebuilds the serving artifacts and the Docker image. Since MovieLens is a fixed dataset with a fixed snapshot date, this demonstrates the **mechanics** of drift-triggered retraining, not a live system.

---

## 10. Dashboard

`dashboards/app.py` (Streamlit) has three tabs:
- **Results:** final test numbers and all validation runs, read from MLflow
- **Drift:** monthly PSI lines with thresholds
- **Try it:** enter a user ID, or pick movies as a new user, and see recommendations live

---

## 11. Engineering: reproducibility, tests, CI

- **Environment:** `uv` with a lockfile (`uv.lock`), Python 3.11. `uv sync` rebuilds the exact environment.
- **Config:** paths, seeds, split dates and thresholds live in `configs/base.yaml`.
- **Experiment tracking:** every run is logged to MLflow with parameters, metrics (split by returning/new users), per-user scores and the git commit.
- **Reproducibility:** fixed seeds; rerunning a training run gives identical scores.
- **Tests:** 79 pytest tests on tiny hand-made data, so they run anywhere without the dataset. They cover metrics against hand-computed values, split boundaries to the second, leakage guards, model behavior on toy data, statistics on cases with known answers, drift on known shifts, and every API path.
- **CI:** GitHub Actions runs ruff and all tests on every push.

---

## 12. How to run it yourself

**Requirements:** macOS or Linux, [uv](https://docs.astral.sh/uv/), Java 17 (for Spark), and on macOS `brew install libomp` (for LightGBM). Docker if you want the container. About 2 GB of free disk space.

```bash
git clone https://github.com/RrHemanth/movielens-recsys.git
cd movielens-recsys
uv sync
uv run pytest -q
```

**Data (about 2 minutes):**
```bash
uv run python scripts/download_data.py        # download + checksum
uv run python scripts/convert_to_parquet.py   # CSV -> typed parquet
uv run python scripts/build_db.py             # DuckDB views
uv run python scripts/make_split.py           # train / val / test by time
uv run python scripts/run_sql.py sql/eda/*.sql
```

**Models on validation (2018):**
```bash
uv run python scripts/eval_popularity.py
uv run python scripts/eval_als.py --train-start-year 2017 --alpha 10 --reg 1.0
uv run python scripts/export_als_vectors.py
uv run python scripts/benchmark_faiss.py
for d in 2017-01-01 2018-01-01; do
  uv run python scripts/build_candidates.py --as-of $d
  uv run python scripts/build_features.py --as-of $d
done
uv run python scripts/eval_reranker.py
uv run python scripts/eval_cold_start.py
uv run python scripts/compare_models_stats.py
uv run python scripts/compare_runs.py
```

**Final test (2019):**
```bash
uv run python scripts/build_candidates.py --as-of 2019-01-01
uv run python scripts/build_features.py --as-of 2019-01-01
uv run python scripts/final_test.py
```

**Serving, monitoring, dashboard:**
```bash
uv run python scripts/build_serving_artifacts.py
uv run uvicorn recsys.serving.app:create_app --factory --port 8000   # docs at http://127.0.0.1:8000/docs

docker build -t movielens-recsys:2019-01-01 .
docker run --rm -p 8000:8000 movielens-recsys:2019-01-01

uv run python scripts/drift_replay.py
uv run streamlit run dashboards/app.py
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## 13. Repository layout

```
├── configs/base.yaml          settings: paths, seed, split dates, thresholds
├── sql/eda/                   exploration queries (DuckDB)
├── src/recsys/
│   ├── data/                  download, parquet conversion, DuckDB, time split, periods
│   ├── features/              re-ranker features + leakage guard
│   ├── models/                popularity, ALS (Spark), similarity, re-ranker, cold start
│   ├── evaluation/            metrics, evaluate(), bootstrap + significance tests
│   ├── monitoring/            PSI / KS drift, retraining rule
│   ├── serving/               FastAPI app
│   ├── candidates.py          candidate generation + candidate recall
│   ├── spark.py, tracking.py  Spark session, MLflow helpers
├── scripts/                   thin entry points for every step
├── tests/                     79 tests on tiny fake data
├── dashboards/app.py          Streamlit dashboard
├── Dockerfile, .dockerignore  serving image
└── .github/workflows/         ci.yml (tests on push), retrain.yml (drift-triggered retraining)
```

The data itself (`data/`), MLflow records and the virtual environment are not in git; everything can be rebuilt from the scripts.

---

## 14. Limitations and honest caveats

- **Offline only.** All numbers come from historical data. Offline gains don't guarantee online gains; I never ran a live test.
- **MovieLens quirks.** Every user has at least 20 ratings, and new users rate many famous movies right after joining, which makes popularity look very strong for them. That's a property of how the data was collected.
- **The shortlist is the ceiling.** It contains only about 32% of the movies users go on to like; the re-ranker can't recommend anything outside it.
- **Accuracy vs. coverage.** The re-ranker is the most accurate model but covers less of the catalog than ALS alone.
- **The one-year ALS window** leaves about 22% of returning users without a vector; they get popularity-only shortlists.
- **Inactive users** get the global popular list without filtering movies they rated years ago.
- **Fold-in** was evaluated with 5 onboarding ratings; I didn't measure how it does with fewer.
- **Spark and FAISS** weren't necessary at this scale. I used them to learn them, and I measured rather than assumed their trade-offs.
- **Retraining** runs on a fixed dataset with a fixed snapshot date, so it shows the mechanics, not a live loop.
- **Tuning on validation** makes validation numbers slightly optimistic, which is why the test year was held out and used once.

---

## 15. What I would do next

- Add more candidate sources (for example, neighbors of a user's recent likes) to raise the 32% ceiling.
- Measure how many onboarding ratings a new user needs before fold-in beats popularity.
- Try a proper ALS fold-in (solving the small least-squares problem) instead of averaging.
- Use content features (genres, tag genome) for movies with no ratings.
- Build the offline replay / interleaving simulation to compare rankers.
- Roll the snapshot date forward in the retraining workflow, so each run trains on the newest data.

---

## 16. Acknowledgements

Data: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems 5(4): 19:1–19:19. https://doi.org/10.1145/2827872. The data is not included in this repository; the download script fetches it from GroupLens and verifies the checksum.
