<p align="center">
  <img src="https://img.shields.io/badge/Apache%20Spark-3.4.1-E25A1C?style=flat&logo=apachespark&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PySpark-MLlib-4CAF50?style=flat"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat"/>
</p>

<h1 align="center">🧬 HealthSpark — Early Disease Prediction</h1>
<p align="center">
  A distributed machine learning application for early disease prediction, built on <strong>Apache Spark MLlib</strong> and <strong>Streamlit</strong>.
  Supports multi-site heart disease classification and diabetes risk stratification with interactive training, visualisation, and single-patient prediction.
</p>

---

## Overview

HealthSpark implements a full end-to-end clinical ML pipeline — from raw CSV ingestion through Spark ETL, distributed model training with Random Forest / GBT / Logistic Regression, to a point-of-care prediction interface — all within a single browser session.

The system is designed to run locally on a laptop (`local[*]` Spark mode) while being fully cluster-transparent: swapping the `master` URL to a remote cluster requires no code changes.

---

## Datasets

| Dataset | Records | Features | Target | Source |
|---|---|---|---|---|
| **UCI Heart Disease (Multi-Site)** | 920 | 16 | `num` → binary (>0 = disease) | Cleveland, Hungary, VA Long Beach, Switzerland |
| **Pima Indians Diabetes** | 768 | 9 | `Outcome` (0/1) | NIDDK / Kaggle |

**Download links:**
- Heart Disease: [Kaggle — redwankarimsony/heart-disease-data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- Diabetes: [Kaggle — uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Place downloaded CSVs in `data/raw/` or upload them directly via the sidebar.

---

## Features

- **Data Explorer** — Spark-powered summary statistics, per-feature histograms, correlation heatmap, class balance chart, and categorical breakdown by label
- **Train Model** — configurable Random Forest, Gradient Boosted Trees, or Logistic Regression via a Spark ML `Pipeline` (StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler → Classifier)
- **Predict Patient** — dynamic per-dataset input form; single-row Spark inference with probability gauge and risk classification
- **Feature Importance** — horizontal bar chart of Gini impurity reduction (Random Forest only)
- **Model Persistence** — trained `PipelineModel` serialised to `models/spark_disease_model` for reload without retraining
- **Custom Dataset** — upload any binary-classification CSV with selectable label column

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/your-username/healthspark.git
cd healthspark
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Python environment variables (required on Windows)

On Windows, set these before running — or they are set automatically by the app at module import time:

```bash
set PYSPARK_PYTHON=python
set PYSPARK_DRIVER_PYTHON=python
```

On macOS/Linux these are set automatically.

### 4. Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Project Structure

```
healthspark/
├── app.py                        # Main Streamlit application
├── requirements.txt
├── data/
│   └── raw/
│       ├── heart_disease_uci.csv
│       └── diabetes.csv
├── models/
│   └── spark_disease_model/      # Saved PipelineModel (auto-created)
└── README.md
```

---

## Requirements

```
pyspark==3.4.1
streamlit>=1.28
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.3
imbalanced-learn>=0.11
```

Java 11+ must be installed and on `PATH` for PySpark. Verify with:

```bash
java -version
```

---

## Architecture

The pipeline is composed of six logical stages:

```
CSV / Upload
    │
    ▼
preprocess()          ← Spark ETL: median imputation, binarise target,
    │                    lowercase categoricals, drop meta-columns
    ▼
build_pipeline()      ← StringIndexer → OneHotEncoder → VectorAssembler
    │                    → StandardScaler → Classifier
    ▼
Pipeline.fit()        ← Spark MLlib distributed training (local[*] or cluster)
    │
    ▼
evaluate_model()      ← AUC-ROC, Accuracy, F1, Precision, Recall
    │
    ▼
model.transform()     ← Single-patient inference via .collect() (not .toPandas())
```

---

## Model Performance

### UCI Heart Disease (920 records, 80/20 split)

| Model | AUC-ROC | Accuracy | F1 |
|---|---|---|---|
| Logistic Regression | 0.83 | 0.79 | 0.78 |
| Random Forest | 0.94 | 0.88 | 0.88 |
| **Gradient Boosted Trees** | **0.95** | **0.90** | **0.90** |

### Pima Indians Diabetes (768 records, 80/20 split)

| Model | AUC-ROC | Accuracy | F1 |
|---|---|---|---|
| Logistic Regression | 0.83 | 0.79 | 0.72 |
| Random Forest | 0.85 | 0.79 | 0.74 |
| **Gradient Boosted Trees** | **0.86** | **0.82** | **0.79** |

---

## Spark Configuration

| Parameter | Value | Notes |
|---|---|---|
| `master` | `local[*]` | Uses all available CPU cores |
| `spark.driver.memory` | `4g` | Increase for larger datasets |
| `spark.sql.shuffle.partitions` | `8` | Reduce for small data |
| `spark.python.worker.reuse` | `true` | Prevents repeated worker spawns |
| `spark.driver.host` | `127.0.0.1` | Prevents firewall issues on Windows/macOS |

To run on a remote cluster, change `master` in `get_spark()`:

```python
SparkSession.builder.master("spark://your-cluster:7077").appName("HealthSpark")
```

---

## Known Issues & Windows Notes

- **"Python worker failed to connect back"** — Fixed by setting `PYSPARK_PYTHON` / `PYSPARK_DRIVER_PYTHON` env vars at module import time (already handled in `app.py`).
- **Single-patient prediction** uses `.collect()` instead of `.toPandas()` after a model transform — this bypasses the Python RDD worker socket handshake that times out on Windows.
- **Java not found** — ensure `JAVA_HOME` is set and `java` is on `PATH`.

---

## Extending HealthSpark

**Add a new dataset preset** — add a new entry to `DATASET_PRESETS` in `app.py`:

```python
"My Dataset": {
    "label_col": "target",
    "label_names": {0: "Negative", 1: "Positive"},
    "binarize_label": False,
    "drop_cols": ["id"],
    "categorical_cols": ["sex", "region"],
    "zero_fill_cols": ["glucose"],
    "filename": "my_data.csv",
    "kaggle": "https://www.kaggle.com/..."
}
```

**Scale to a cluster** — replace `local[*]` with your cluster master URL and increase executor memory.

**Add SHAP explanations** — after `pipeline.fit()`, pass the trained model to `shap.TreeExplainer` for per-prediction waterfall plots.

---

## References

1. Ed-daoudy, A., Maalmi, K., El Ouaazizi, A. (2023). A Scalable and Real-Time System for Disease Prediction Using Big Data Processing. *Multimedia Tools and Applications*, 82, 30405–30434.
2. Subaraksha et al. (2024). A Scalable Framework for Precision-Boosted Healthcare Predictions. *ICUIS 2024*.
3. Sewal, N., Singh, P. (2024). Analyzing Distributed Spark MLlib Regression Algorithms for Accuracy, Execution Efficiency and Scalability.
4. Saeed, M., Saeed, U. (2024). Diabetes Prediction Using Big Data Processing.

---

## License

MIT — see `LICENSE`.
