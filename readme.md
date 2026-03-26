# 🎬 Clickstream Analytics & ALS Recommendation System

Real-time big data analytics pipeline built on MovieLens 100K, combining probabilistic 
streaming algorithms (Flajolet-Martin, Count-Min Sketch) with a Spark MLlib ALS 
recommendation engine, visualized through a live Streamlit dashboard.

---

## What this project does

Simulates a production-style clickstream analytics pipeline where user rating events 
stream in as batches (like Kafka), are processed by probabilistic data structures for 
memory-efficient counting, and feed a collaborative filtering model that generates 
personalized movie recommendations.

---

## Architecture
```
MovieLens 100K (100K ratings · 943 users · 1682 movies)
        ↓
PySpark Session — sliding window aggregation, HyperLogLog
        ↓
┌─────────────────┬──────────────────┬─────────────────┐
│ Flajolet-Martin │ Count-Min Sketch │   ALS (MLlib)   │
│ Unique user     │ Item frequency   │ Collaborative   │
│ cardinality     │ estimation       │ filtering       │
└─────────────────┴──────────────────┴─────────────────┘
        ↓
CSV exports (kpi · metrics · event_counts · time_series · trending · recommendations)
        ↓
Streamlit Dashboard (auto-refreshes every 2–10s)
```

---

## Algorithms

**Flajolet-Martin (1985)**
Estimates unique user count in O(1) memory using 64 hash functions across 8 groups.
Uses bitmap-based trailing zeros with PHI=0.7713 correction factor exactly as specified
in the original paper. Median-of-means grouping reduces variance. Achieves ~5-12% error
vs exact counting, using kilobytes instead of storing every user ID.

**Count-Min Sketch (Cormode & Muthukrishnan, 2005)**
Estimates item interaction frequencies using a fixed 500×7 table (~14 KB total).
Uses MD5 hashing for cross-session reproducibility (Python's built-in hash() randomizes
seeds at runtime). Always overestimates, never underestimates. Benchmarked across
widths 50→2000 for memory/accuracy tradeoff analysis.

**Simulated Kafka Stream**
100K events processed in 1000-event batches with stateful FM and CMS updating
incrementally across 100 batches — mirrors a real Spark Structured Streaming + Kafka
consumer pattern.

**ALS (Alternating Least Squares)**
Matrix factorization on the 943×1682 user-item rating matrix (sparsity ~94%).
Trained on 80K ratings, evaluated on 20K held-out test split.
- Rank: 20 latent factors
- Regularization: 0.1
- Iterations: 10
- Cold start strategy: drop

---

## Results

| Algorithm | Metric | Value |
|---|---|---|
| Flajolet-Martin | Estimation error | ~11.88% |
| Count-Min Sketch | Avg error (top 20 items) | 5.63% |
| Count-Min Sketch | Memory vs exact counting | 6.7x reduction (13.7 KB vs 91.1 KB) |
| ALS | RMSE (test set) | 0.9195 |
| ALS | Precision@10 | 3.96% |
| ALS | Recall@10 | 4.01% |

> Note: Precision/Recall values are typical for collaborative filtering on sparse matrices.
> The MovieLens 100K matrix is ~94% empty — low absolute precision is expected and
> consistent with published benchmarks on this dataset.

---

## Tech Stack

- **Python 3.x**
- **PySpark** — distributed processing, sliding window aggregation, ALS
- **pandas / numpy** — data manipulation
- **Streamlit** — live dashboard
- **streamlit-autorefresh** — periodic data reload
- **matplotlib / seaborn** — notebook visualizations
- **Google Colab** — notebook execution environment

---

## Project Structure
```
├── index.ipynb                    # Main notebook — run this first
└── Dashboard files/
    ├── app.py                     # Streamlit dashboard
    ├── kpi.csv                    # Total users, events, avg rating, FM estimate
    ├── metrics.csv                # RMSE, Precision@10, Recall@10
    ├── event_counts.csv           # view / addtocart / transaction counts
    ├── time_series.csv            # Windowed event counts over time
    ├── trending.csv               # All 1682 movies ranked by interaction count
    └── recommendations.csv        # Top 10 ALS recommendations per user (9430 rows)
```

---

## How to Run

### Step 1 — Run the notebook (Google Colab recommended)

Open `index.ipynb` in Google Colab and run all cells in order.
Cell 24 exports all 6 CSV files. Download and place them in the same
folder as `app.py`.

The notebook will:
1. Download MovieLens 100K automatically
2. Install PySpark
3. Run FM, CMS, and Kafka stream simulation
4. Train ALS model
5. Export all CSVs

### Step 2 — Run the dashboard locally
```bash
pip install streamlit streamlit-autorefresh pandas numpy
streamlit run app.py
```

Make sure all 6 CSV files are in the same directory as `app.py`.

---

## Dataset

**MovieLens 100K** — F. Maxwell Harper and Joseph A. Konstan (2015).
100,000 ratings from 943 users on 1682 movies collected through the MovieLens
website. Ratings are on a 1–5 star scale.

Downloaded automatically by the notebook from:
`http://files.grouplens.org/datasets/movielens/ml-100k.zip`

Ratings mapped to clickstream events:
- 1–3 stars → `view`
- 4 stars → `addtocart`  
- 5 stars → `transaction`

---

## References

1. Flajolet, P. & Martin, G. (1985). Probabilistic Counting Algorithms for Data Base Applications. *Journal of Computer and System Sciences.*
2. Cormode, G. & Muthukrishnan, S. (2005). An Improved Data Stream Summary: The Count-Min Sketch and its Applications. *Journal of Algorithms.*
3. Koren, Y., Bell, R. & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer.*
4. Harper, F. & Konstan, J. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems.*