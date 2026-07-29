# 🎬 Clickstream Analytics & ALS Recommendation System

A big-data analytics and recommendation-system prototype built using the **MovieLens 100K dataset**.

The project converts movie-rating records into simulated clickstream events, processes them using **PySpark**, applies memory-efficient probabilistic streaming algorithms such as **Flajolet-Martin** and **Count-Min Sketch**, trains an **ALS collaborative-filtering model**, and presents the generated analytics through an auto-refreshing **Streamlit dashboard**.

---

## 📌 Project Overview

This project demonstrates how large volumes of user-interaction events can be processed without storing every individual count directly in memory.

MovieLens ratings are converted into three simulated clickstream event types:

* Ratings from **1 to 3** → `view`
* Rating of **4** → `addtocart`
* Rating of **5** → `transaction`

The resulting events are processed through:

1. PySpark window-based aggregation
2. HyperLogLog-based approximate counting
3. Flajolet-Martin unique-user estimation
4. Count-Min Sketch item-frequency estimation
5. Simulated Kafka-style batch processing
6. Spark MLlib ALS recommendation generation
7. CSV export
8. Streamlit dashboard visualization

> This is an educational prototype that simulates a production-style streaming pipeline. It does not currently connect to a real Kafka broker or a live Spark Structured Streaming source.

---

## 🏗️ System Architecture

```text
MovieLens 100K Dataset
100,000 ratings · 943 users · 1,682 movies
                         │
                         ▼
              Data Loading and Preprocessing
       Rating events converted into clickstream events
                         │
                         ▼
                 PySpark DataFrame
          10-minute windows with a 5-minute slide
                         │
       ┌─────────────────┼────────────────────┐
       ▼                 ▼                    ▼
Flajolet-Martin   Count-Min Sketch      Spark MLlib ALS
Unique-user       Item-frequency        Collaborative
estimation        estimation            filtering
       │                 │                    │
       └─────────────────┼────────────────────┘
                         ▼
              Simulated Batch Processing
         1,000 events per batch · 100 batches
                         │
                         ▼
                    CSV Exports
     KPI · metrics · event counts · time series
       trending items · user recommendations
                         │
                         ▼
                Streamlit Dashboard
          Configurable 2–10 second auto-refresh
```

---

## ✨ Main Features

* Processes all **100,000 MovieLens ratings**
* Handles **943 users** and **1,682 movies**
* Converts ratings into simulated clickstream events
* Performs PySpark sliding-window aggregation
* Uses Spark's `approx_count_distinct` for HyperLogLog-based counting
* Implements Flajolet-Martin from scratch
* Implements Count-Min Sketch from scratch
* Simulates Kafka-style incremental batch processing
* Trains an ALS collaborative-filtering model
* Produces top-10 recommendations for every user
* Exports analysis results into six CSV files
* Displays results through an interactive Streamlit dashboard
* Supports configurable automatic dashboard refresh

---

## 🧠 Algorithms

### 1. Flajolet-Martin Algorithm

Flajolet-Martin is used to estimate the number of unique users without storing every user ID in a set.

The implementation uses:

* 64 pairwise-independent hash functions
* 8 groups with 8 hash functions per group
* 32-position bitmaps
* Trailing-zero analysis
* Correction factor `PHI = 0.77130`
* Group averaging followed by median aggregation
* Fixed random seed for reproducibility

The algorithm provides constant-size memory usage relative to the number of processed events.

#### Current result

```text
Exact unique users:     943
FM estimated users:     831
Estimation error:       11.88%
```

---

### 2. Count-Min Sketch

Count-Min Sketch is used to estimate how frequently each movie appears in the interaction stream.

Instead of maintaining a separate exact counter for every item, it stores counts in a fixed-size two-dimensional integer table.

The implementation uses:

* Width: `500`
* Depth: `7`
* Table size: `500 × 7`
* Integer type: `int32`
* MD5-based deterministic hashing
* Minimum value across hash rows for querying

Count-Min Sketch may overestimate an item's frequency because of hash collisions, but it does not underestimate it.

The notebook also benchmarks widths:

```text
50, 100, 200, 500, 1000 and 2000
```

This demonstrates the trade-off between memory consumption and estimation accuracy.

#### Current result

```text
Average error for top 20 items:  5.63%
Count-Min Sketch memory:         13.7 KB
Exact dictionary memory:         91.1 KB
Memory reduction:                6.7×
```

---

### 3. Simulated Kafka-Style Stream

The project processes the dataset in batches to demonstrate how stateful streaming algorithms behave as more events arrive.

Configuration:

```text
Total events:       100,000
Events per batch:   1,000
Total batches:      100
```

During each batch, the system incrementally updates:

* Exact unique-user count
* Flajolet-Martin user estimate
* Exact item frequencies
* Count-Min Sketch frequencies
* Event-type counts

This reproduces the logical flow of:

```text
Kafka Producer → Stream Consumer → Stateful Analytics
```

However, Kafka itself is not currently installed or connected.

---

### 4. ALS Recommendation System

The recommendation engine uses **Alternating Least Squares**, provided by Spark MLlib.

ALS factorizes the user-item rating matrix into lower-dimensional user and item representations.

#### Model configuration

```text
Rank:                  20 latent factors
Regularization:        0.1
Maximum iterations:    10
Training split:        80%
Testing split:         20%
Cold-start strategy:   drop
Preference type:       explicit ratings
Random seed:           42
```

#### Actual split from the current run

```text
Training records:      80,062
Testing records:       19,938
```

Relevant test items are defined as movies with ratings greater than or equal to four.

The trained model generates the top 10 recommendations for every user.

---

## 📊 Current Results

| Component             | Metric                    |  Result |
| --------------------- | ------------------------- | ------: |
| Dataset               | Total events              | 100,000 |
| Dataset               | Unique users              |     943 |
| Dataset               | Unique movies             |   1,682 |
| Flajolet-Martin       | Estimated users           |     831 |
| Flajolet-Martin       | Estimation error          |  11.88% |
| Count-Min Sketch      | Average top-20 error      |   5.63% |
| Count-Min Sketch      | Memory usage              | 13.7 KB |
| Exact item dictionary | Memory usage              | 91.1 KB |
| Count-Min Sketch      | Memory reduction          |    6.7× |
| ALS                   | Test RMSE                 |  0.9205 |
| ALS                   | Precision@10              |   3.90% |
| ALS                   | Recall@10                 |   3.98% |
| ALS                   | Generated recommendations |   9,430 |

### Recommendation-result interpretation

The ALS model achieves an RMSE of approximately `0.92`, meaning its predicted rating differs from the actual rating by roughly 0.92 rating points on average.

Precision@10 and Recall@10 are modest. This means only a limited number of held-out liked movies appear in each user's top-10 list.

The project successfully demonstrates the complete recommendation pipeline, but the current recommendation quality should not be considered production-grade. It could be improved through:

* ALS hyperparameter tuning
* Ranking-based model evaluation
* Better train-test splitting
* Removing already-rated items from recommendations
* Implicit-feedback modelling
* Hybrid content and collaborative filtering
* Additional user and movie features

---

## 📈 Event Distribution

| Event         |       Count |
| ------------- | ----------: |
| `view`        |      44,625 |
| `addtocart`   |      34,174 |
| `transaction` |      21,201 |
| **Total**     | **100,000** |

---

## 🛠️ Technology Stack

| Technology            | Usage                                              |
| --------------------- | -------------------------------------------------- |
| Python                | Main programming language                          |
| PySpark               | Distributed processing, window aggregation and ALS |
| Spark MLlib           | Collaborative-filtering recommendation model       |
| pandas                | Data loading, preprocessing and CSV export         |
| NumPy                 | Numerical arrays and probabilistic data structures |
| Streamlit             | Interactive analytics dashboard                    |
| streamlit-autorefresh | Configurable dashboard refreshing                  |
| Matplotlib            | Algorithm-performance visualizations               |
| Seaborn               | Notebook visualization styling                     |
| Google Colab          | Recommended notebook execution environment         |
| Jupyter Notebook      | Main analytics and model-development environment   |

---

## 📁 Project Structure

```text
Click-Stream-Analysis-/
│
├── README.md
├── .gitignore
│
├── Dataset/
│   ├── ml-100k.zip
│   └── ml-100k/
│
├── Source Code/
│   ├── index.ipynb
│   │
│   └── Dashboard files/
│       ├── app.py
│       ├── kpi.csv
│       ├── metrics.csv
│       ├── event_counts.csv
│       ├── time_series.csv
│       ├── trending.csv
│       ├── recommendations.csv
│       │
│       └── Dashboard Sample View/
│           ├── 1.png
│           ├── 2.png
│           ├── 3.png
│           └── 4.png
│
├── Report/
│   └── IEEE_Report.docx
│
├── Slide/
│   └── 123102057 - 56.pptx
│
└── Software/
    └── readme.txt.txt
```

---

## 📄 Generated CSV Files

| File                  | Description                                         |
| --------------------- | --------------------------------------------------- |
| `kpi.csv`             | Total users, events, average rating and FM results  |
| `metrics.csv`         | RMSE, Precision@10 and Recall@10                    |
| `event_counts.csv`    | Counts of view, add-to-cart and transaction events  |
| `time_series.csv`     | Event counts across sliding time windows            |
| `trending.csv`        | All 1,682 movies ordered by exact interaction count |
| `recommendations.csv` | Top-10 ALS recommendations for all 943 users        |

Current output sizes:

```text
kpi.csv:                 5 rows
metrics.csv:             3 rows
event_counts.csv:        3 rows
time_series.csv:        27,927 rows
trending.csv:            1,682 rows
recommendations.csv:     9,430 rows
```

---

## 🚀 How to Run

### Method 1: Run the existing dashboard

The repository already contains the generated CSV files, so the dashboard can be run without executing the notebook again.

#### 1. Clone the repository

```powershell
git clone https://github.com/shubham-alyan24/Click-Stream-Analysis-.git
cd "Click-Stream-Analysis-"
```

#### 2. Create a virtual environment

```powershell
python -m venv .venv
```

#### 3. Activate the virtual environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

#### 4. Install dashboard dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install streamlit streamlit-autorefresh pandas numpy
```

#### 5. Open the dashboard directory

```powershell
cd ".\Source Code\Dashboard files"
```

#### 6. Start Streamlit

```powershell
python -m streamlit run app.py
```

The dashboard should open automatically at:

```text
http://localhost:8501
```

All six CSV files must remain in the same directory as `app.py`.

---

### Method 2: Reproduce the complete analytics pipeline

Google Colab is recommended because the notebook installs and configures PySpark automatically.

1. Open `Source Code/index.ipynb`.
2. Upload it to Google Colab.
3. Select **Runtime → Run all**.
4. Allow the notebook to download and extract MovieLens 100K.
5. Wait for the FM, CMS, stream simulation and ALS cells to complete.
6. Run the CSV-export cells.
7. Download `project_files.zip`.
8. Extract the generated CSV files.
9. Replace the CSV files inside:

```text
Source Code/Dashboard files/
```

The notebook performs the following operations:

```text
Download MovieLens
        ↓
Preprocess ratings
        ↓
Create PySpark DataFrame
        ↓
Run sliding-window analysis
        ↓
Run HyperLogLog unique-user estimation
        ↓
Run Flajolet-Martin
        ↓
Run Count-Min Sketch
        ↓
Simulate 100 event batches
        ↓
Train and evaluate ALS
        ↓
Export dashboard CSV files
```

---

## 🖼️ Dashboard Preview

![Clickstream Dashboard](Source%20Code/Dashboard%20files/Dashboard%20Sample%20View/1.png)

Additional dashboard screenshots are available inside:

```text
Source Code/Dashboard files/Dashboard Sample View/
```

---

## 📚 Dataset

The project uses the **MovieLens 100K dataset**.

Dataset details:

```text
Ratings:       100,000
Users:         943
Movies:        1,682
Rating scale:  1–5
```

The dataset was collected through the MovieLens website.

The notebook downloads it from:

```text
http://files.grouplens.org/datasets/movielens/ml-100k.zip
```

A copy of the compressed and extracted dataset is also included inside the repository's `Dataset` directory.

### Citation

F. Maxwell Harper and Joseph A. Konstan.
*The MovieLens Datasets: History and Context.*
ACM Transactions on Interactive Intelligent Systems, 2015.


---

## 🔮 Future Improvements

* Connect an actual Apache Kafka producer and consumer
* Use Spark Structured Streaming for continuous processing
* Store events in PostgreSQL, MongoDB or Cassandra
* Serve recommendations through FastAPI
* Display movie titles, genres and posters
* Remove already-consumed items from recommendations
* Add popularity-based recommendations for cold-start users
* Tune ALS rank, regularization and iteration parameters
* Add implicit-feedback ALS
* Add MAP@K, NDCG@K and hit-rate evaluation
* Build a hybrid collaborative and content-based recommender
* Deploy the Streamlit dashboard online
* Add Docker support
* Add automated testing and CI/CD

---

## 📖 References

1. Flajolet, P. and Martin, G.
   *Probabilistic Counting Algorithms for Data Base Applications.*
   Journal of Computer and System Sciences, 1985.

2. Cormode, G. and Muthukrishnan, S.
   *An Improved Data Stream Summary: The Count-Min Sketch and Its Applications.*
   Journal of Algorithms, 2005.

3. Koren, Y., Bell, R. and Volinsky, C.
   *Matrix Factorization Techniques for Recommender Systems.*
   IEEE Computer, 2009.

4. Harper, F. and Konstan, J.
   *The MovieLens Datasets: History and Context.*
   ACM Transactions on Interactive Intelligent Systems, 2015.

---

## 👤 Author

**Shubham Alyan**

GitHub: [shubham-alyan24](https://github.com/shubham-alyan24)

---

## 📜 License
This project was developed for academic and educational purposes. The MovieLens dataset remains subject to the terms provided by GroupLens Research.

This project was developed for academic and educational purposes. The MovieLens dataset remains subject to the terms provided by GroupLens Research.
