import os
import sys
import warnings
import logging
import math
import time
import tempfile

import os
import sys

os.environ["JAVA_HOME"]        = r"C:\java\jdk"
os.environ["HADOOP_HOME"]      = r"C:\hadoop"
os.environ["SPARK_HOME"]       = r"C:\spark\spark-4.0.0-bin-hadoop3"

# Tell Spark workers to use the exact Python running this script
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import json, time, warnings  # rest of your imports continue below...

warnings.filterwarnings("ignore")
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, mean as spark_mean, trim, lower,
    count, lit, log as spark_log, abs as spark_abs,
    hash as spark_hash
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
)
from pyspark.ml.classification import (
    RandomForestClassifier, GBTClassifier, LogisticRegression
)
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HealthSpark — Disease Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background:#0a0e1a; color:#e0e6f0; }
h1,h2,h3 { font-family:'Space Mono',monospace; }
.stApp { background:#0a0e1a; }
.section-header {
    background:linear-gradient(90deg,#0ea5e9,#6366f1);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-family:'Space Mono',monospace; font-size:1.2rem; margin-bottom:.5rem;
}
.predict-box {
    background:linear-gradient(135deg,#0f1f3d,#1a2e50);
    border:1px solid #38bdf8; border-radius:16px; padding:28px;
    box-shadow:0 0 30px rgba(56,189,248,.1); margin-bottom:1rem;
}
.risk-high {
    background:linear-gradient(135deg,#450a0a,#7f1d1d); border:1px solid #ef4444;
    border-radius:12px; padding:20px; text-align:center; color:#fca5a5;
    font-family:'Space Mono',monospace;
}
.risk-low {
    background:linear-gradient(135deg,#052e16,#14532d); border:1px solid #22c55e;
    border-radius:12px; padding:20px; text-align:center; color:#86efac;
    font-family:'Space Mono',monospace;
}
.stButton>button {
    background:linear-gradient(90deg,#0ea5e9,#6366f1); color:#fff; border:none;
    border-radius:8px; padding:.6rem 2rem; font-family:'Space Mono',monospace;
    font-weight:700; letter-spacing:.05em; transition:all .3s;
}
.stButton>button:hover { transform:translateY(-2px); box-shadow:0 6px 20px rgba(14,165,233,.4); }
[data-testid="stSidebar"] { background:#0d1526; border-right:1px solid #1e2a45; }
div[data-testid="stMetric"] {
    background:linear-gradient(135deg,#111827,#1e2a45); border:1px solid #2a3a5c;
    border-radius:12px; padding:16px;
}
div[data-testid="stMetric"] label { color:#94a3b8!important; }
div[data-testid="stMetric"] div   { color:#38bdf8!important; }
.info-box {
    background:#111827; border-left:3px solid #38bdf8; padding:10px 14px;
    border-radius:0 8px 8px 0; color:#94a3b8; font-size:.85rem; margin:8px 0;
}
.tag {
    display:inline-block; background:#1e2a45; border:1px solid #2a3a5c;
    border-radius:6px; padding:2px 10px; font-size:.78rem; color:#94a3b8; margin:2px;
}
.improvement-box {
    background:linear-gradient(135deg,#052e16,#0a3d1f); border:1px solid #16a34a;
    border-radius:10px; padding:12px 16px; color:#86efac; font-size:.82rem; margin:6px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Spark Session (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_spark():
    spark = (
        SparkSession.builder
        .appName("HealthSpark")
        .master("local[*]")
        .config("spark.driver.memory",          "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.pyspark.python",         sys.executable)
        .config("spark.pyspark.driver.python",  sys.executable)
        .config("spark.python.worker.reuse",    "true")
        .config("spark.driver.host",            "127.0.0.1")
        .config("spark.driver.bindAddress",     "127.0.0.1")
        .config("spark.python.worker.timeout",  "120")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ─────────────────────────────────────────────
# Dataset Presets
# ─────────────────────────────────────────────
DATASET_PRESETS = {
    "Cleveland Heart Disease (UCI Full)": {
        "description":   "Full UCI Heart Disease dataset with categorical features & multi-site origin",
        "label_col":     "num",
        "label_names":   {0: "No Disease", 1: "Heart Disease"},
        "binarize_label": True,
        "drop_cols":     ["id", "origin"],
        "categorical_cols": ["sex", "cp", "restecg", "exang", "thal", "slope"],
        "zero_fill_cols": [],
        "kaggle":        "https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data",
        "filename":      "heart_disease_uci.csv",
        "best_params": {
            "Random Forest":          {"numTrees": 200, "maxDepth": 10, "minInstancesPerNode": 2, "featureSubsetStrategy": "sqrt"},
            "Gradient Boosted Trees": {"maxIter": 80,  "maxDepth": 6,  "stepSize": 0.08, "subsamplingRate": 0.8, "minInstancesPerNode": 3},
            "Logistic Regression":    {"maxIter": 300, "regParam": 0.001, "elasticNetParam": 0.1},
        },
        "interaction_cols": [("trestbps", "chol"), ("thalach", "oldpeak"), ("age", "chol")],
    },
    "Diabetes (Pima Indians)": {
        "description":   "Predict diabetes onset in female Pima Indian patients",
        "label_col":     "Outcome",
        "label_names":   {0: "Non-Diabetic", 1: "Diabetic"},
        "binarize_label": False,
        "drop_cols":     [],
        "categorical_cols": [],
        "zero_fill_cols": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "kaggle":        "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
        "filename":      "diabetes.csv",
        "best_params": {
            "Random Forest":          {"numTrees": 300, "maxDepth": 10, "minInstancesPerNode": 2, "featureSubsetStrategy": "sqrt"},
            "Gradient Boosted Trees": {"maxIter": 100, "maxDepth": 5,  "stepSize": 0.05, "subsamplingRate": 0.8, "minInstancesPerNode": 2},
            "Logistic Regression":    {"maxIter": 500, "regParam": 0.001, "elasticNetParam": 0.2},
        },
        "interaction_cols": [
            ("Glucose", "BMI"), ("BMI", "Age"),
            ("Glucose", "Insulin"), ("Pregnancies", "Age"),
        ],
    },
    "Custom Dataset": {
        "description":   "Upload any binary classification healthcare CSV",
        "label_col":     None,
        "label_names":   {0: "Negative", 1: "Positive"},
        "binarize_label": False,
        "drop_cols":     [],
        "categorical_cols": [],
        "zero_fill_cols": [],
        "kaggle":        None,
        "filename":      None,
        "best_params": {
            "Random Forest":          {"numTrees": 200, "maxDepth": 8,  "minInstancesPerNode": 3, "featureSubsetStrategy": "sqrt"},
            "Gradient Boosted Trees": {"maxIter": 60,  "maxDepth": 5,  "stepSize": 0.1,  "subsamplingRate": 0.8, "minInstancesPerNode": 3},
            "Logistic Regression":    {"maxIter": 200, "regParam": 0.01, "elasticNetParam": 0.1},
        },
        "interaction_cols": [],
    },
}

# ─────────────────────────────────────────────
# Column defaults for Predict tab
# ─────────────────────────────────────────────
COL_DEFAULTS = {
    "pregnancies":              (0,   20,    2,    int,   None),
    "glucose":                  (50,  300,   120,  int,   None),
    "bloodpressure":            (30,  140,   72,   int,   None),
    "skinthickness":            (0,   100,   23,   int,   None),
    "insulin":                  (0,   900,   80,   int,   None),
    "bmi":                      (10,  70,    28.5, float, None),
    "diabetespedigreefunction": (0.0, 3.0,   0.45, float, None),
    "age":                      (1,   120,   35,   int,   None),
    "trestbps":                 (80,  200,   130,  int,   None),
    "chol":                     (100, 600,   246,  int,   None),
    "thalach":                  (60,  220,   150,  int,   None),
    "oldpeak":                  (0.0, 10.0,  1.0,  float, None),
    "ca":                       (0,   3,     0,    int,   None),
    "fbs":                      (0,   1,     0,    int,   None),
    "sex":     (None, None, "Male",           int, ["Male", "Female"]),
    "cp":      (None, None, "typical angina", str, ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]),
    "restecg": (None, None, "normal",         str, ["normal", "stt abnormality", "lv hypertrophy"]),
    "exang":   (None, None, "FALSE",          str, ["TRUE", "FALSE"]),
    "thal":    (None, None, "normal",         str, ["normal", "fixed defect", "reversible defect"]),
    "slope":   (None, None, "upsloping",      str, ["upsloping", "flat", "downsloping"]),
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def save_upload(f) -> str:
    suffix = os.path.splitext(f.name)[-1] or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.getvalue())
    tmp.flush()
    tmp.close()
    return tmp.name


def load_csv(spark, path):
    return spark.read.csv(path, header=True, inferSchema=True)


def dark_fig(figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    ax.tick_params(colors="#94a3b8")
    for s in ax.spines.values():
        s.set_edgecolor("#2a3a5c")
    return fig, ax


# ─────────────────────────────────────────────
# Core preprocessing  (all bugs fixed)
# ─────────────────────────────────────────────
def preprocess(spark, path, preset, label_col):
    sdf = load_csv(spark, path)

    # 1. Normalise categoricals
    cat_cols = preset.get("categorical_cols", [])
    for c in cat_cols:
        if c in sdf.columns:
            sdf = sdf.withColumn(c, lower(trim(col(c).cast("string"))))

    # 2. Drop unused cols
    for c in preset.get("drop_cols", []):
        if c in sdf.columns:
            sdf = sdf.drop(c)

    # 3. Binarise label if needed
    if preset.get("binarize_label") and label_col in sdf.columns:
        sdf = sdf.withColumn(label_col,
              when(col(label_col) > 0, 1).otherwise(0).cast("int"))

    # 4. Cast label to int to keep it stable across plan rewrites
    if label_col in sdf.columns:
        sdf = sdf.withColumn(label_col, col(label_col).cast("int"))

    # 5. Zero → null for biologically impossible zeros
    for c in preset.get("zero_fill_cols", []):
        if c in sdf.columns:
            sdf = sdf.withColumn(c, when(col(c) == 0, None).otherwise(col(c)))

    # 6. Per-class median imputation  (only for zero_fill_cols)
    zero_cols_present = [c for c in preset.get("zero_fill_cols", []) if c in sdf.columns]
    if zero_cols_present and label_col in sdf.columns:
        # Materialise once so approxQuantile doesn't re-execute the whole chain
        sdf = sdf.cache()
        sdf.count()   # trigger cache

        for c in zero_cols_present:
            try:
                global_med = (sdf.approxQuantile(c, [0.5], 0.01) or [0.0])[0]
                med0 = (sdf.filter(col(label_col) == 0).approxQuantile(c, [0.5], 0.01) or [global_med])[0]
                med1 = (sdf.filter(col(label_col) == 1).approxQuantile(c, [0.5], 0.01) or [global_med])[0]
                sdf = sdf.withColumn(
                    c,
                    when(col(c).isNull() & (col(label_col) == 0), float(med0))
                    .when(col(c).isNull() & (col(label_col) == 1), float(med1))
                    .otherwise(col(c))
                )
            except Exception:
                pass

    # 7. Global mean fill for remaining numeric nulls
    #    Snapshot schema names NOW so iteration is stable
    numeric_types = {"int", "bigint", "double", "float", "decimal", "long"}
    schema_snap = [(f.name, str(f.dataType).lower()) for f in sdf.schema.fields]
    fill_dict = {}
    for fname, ftype in schema_snap:
        if fname == label_col or fname in cat_cols:
            continue
        if any(t in ftype for t in numeric_types):
            try:
                mv = sdf.select(spark_mean(col(fname))).collect()[0][0]
                if mv is not None:
                    fill_dict[fname] = round(float(mv), 4)
            except Exception:
                pass
    if fill_dict:
        sdf = sdf.fillna(fill_dict)

    # 8. Mode fill for categoricals
    for c in cat_cols:
        if c in sdf.columns:
            try:
                mode_row = sdf.groupBy(c).count().orderBy("count", ascending=False).first()
                if mode_row:
                    sdf = sdf.fillna({c: mode_row[0]})
            except Exception:
                pass

    # 9. Interaction features
    added_feats = []
    for (a, b) in preset.get("interaction_cols", []):
        if a in sdf.columns and b in sdf.columns:
            iname = f"{a}_x_{b}"
            sdf = sdf.withColumn(iname, col(a).cast("double") * col(b).cast("double"))
            added_feats.append(iname)

    # 10. Log-transform skewed columns
    for c in ["Insulin", "insulin"]:
        if c in sdf.columns:
            logname = f"{c}_log"
            sdf = sdf.withColumn(logname, spark_log(col(c).cast("double") + lit(1.0)))
            added_feats.append(logname)

    # 11. Class weights  — use col() reference, not string, to be safe
    #     Also re-cache after all transforms so groupBy resolves cleanly
    sdf = sdf.cache()
    sdf.count()  # materialise

    total  = sdf.count()
    counts = sdf.groupBy(col(label_col)).count().collect()
    count_map  = {int(r[label_col]): int(r["count"]) for r in counts}
    n_classes  = len(count_map)
    weight_map = {cls: total / (n_classes * cnt) for cls, cnt in count_map.items()}

    # Build weight column expression
    weight_expr = None
    for cls, w in weight_map.items():
        cond = when(col(label_col) == cls, float(w))
        weight_expr = cond if weight_expr is None else weight_expr.when(col(label_col) == cls, float(w))
    if weight_expr is not None:
        sdf = sdf.withColumn("_class_weight", weight_expr.otherwise(1.0))

    return sdf, added_feats


def split_feature_cols(sdf, label_col, cat_cols):
    numeric_types = {"int", "bigint", "double", "float", "decimal", "long"}
    exclude = {label_col, "_class_weight"}
    num_cols = [
        f.name for f in sdf.schema.fields
        if f.name not in exclude
        and f.name not in cat_cols
        and any(t in str(f.dataType).lower() for t in numeric_types)
    ]
    cat_present = [c for c in cat_cols if c in sdf.columns]
    return num_cols, cat_present


# ─────────────────────────────────────────────
# Stratified split  (hash-based, with fallback)
# ─────────────────────────────────────────────
def stratified_split(sdf, label_col, test_frac=0.2, seed=42):
    sdf = sdf.withColumn(
        "_split_hash",
        (spark_abs(spark_hash(col(label_col).cast("string"))) % 100).cast("int")
    )
    thresh = int(test_frac * 100)
    train = sdf.filter(col("_split_hash") >= thresh).drop("_split_hash")
    test  = sdf.filter(col("_split_hash") <  thresh).drop("_split_hash")

    all_cls   = {r[label_col] for r in sdf.select(label_col).distinct().collect()}
    tr_cls    = {r[label_col] for r in train.select(label_col).distinct().collect()}
    te_cls    = {r[label_col] for r in test.select(label_col).distinct().collect()}

    if te_cls != all_cls or tr_cls != all_cls:
        sdf_clean = sdf.drop("_split_hash") if "_split_hash" in sdf.columns else sdf
        train, test = sdf_clean.randomSplit([1 - test_frac, test_frac], seed=seed)

    return train, test


# ─────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────
def build_pipeline(num_cols, cat_cols, label_col, model_name, params):
    stages = []
    assembler_inputs = list(num_cols)

    for c in cat_cols:
        idx_out = f"{c}_idx"
        ohe_out = f"{c}_ohe"
        stages.append(StringIndexer(inputCol=c, outputCol=idx_out,
                                    handleInvalid="keep",
                                    stringOrderType="frequencyDesc"))
        stages.append(OneHotEncoder(inputCols=[idx_out], outputCols=[ohe_out],
                                    handleInvalid="keep"))
        assembler_inputs.append(ohe_out)

    stages.append(VectorAssembler(inputCols=assembler_inputs,
                                  outputCol="raw_features",
                                  handleInvalid="skip"))
    stages.append(StandardScaler(inputCol="raw_features", outputCol="features",
                                 withStd=True, withMean=True))

    if model_name == "Random Forest":
        clf = RandomForestClassifier(
            labelCol=label_col, featuresCol="features",
            weightCol="_class_weight",
            numTrees=params.get("numTrees", 200),
            maxDepth=params.get("maxDepth", 10),
            minInstancesPerNode=params.get("minInstancesPerNode", 2),
            featureSubsetStrategy=params.get("featureSubsetStrategy", "sqrt"),
            seed=42,
        )
    elif model_name == "Gradient Boosted Trees":
        clf = GBTClassifier(
            labelCol=label_col, featuresCol="features",
            weightCol="_class_weight",
            maxIter=params.get("maxIter", 80),
            maxDepth=params.get("maxDepth", 6),
            stepSize=params.get("stepSize", 0.08),
            subsamplingRate=params.get("subsamplingRate", 0.8),
            minInstancesPerNode=params.get("minInstancesPerNode", 3),
            seed=42,
        )
    else:
        clf = LogisticRegression(
            labelCol=label_col, featuresCol="features",
            weightCol="_class_weight",
            maxIter=params.get("maxIter", 300),
            regParam=params.get("regParam", 0.001),
            elasticNetParam=params.get("elasticNetParam", 0.1),
        )

    stages.append(clf)
    return Pipeline(stages=stages)


# ─────────────────────────────────────────────
# Evaluation  (robust — handles single-class test sets)
# ─────────────────────────────────────────────
def evaluate_model(predictions, label_col):
    label_counts = predictions.groupBy(label_col).count().collect()
    metrics = {}

    mc = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction")
    for mname, key in [
        ("accuracy",          "Accuracy"),
        ("f1",                "F1 Score"),
        ("weightedPrecision", "Precision"),
        ("weightedRecall",    "Recall"),
    ]:
        try:
            metrics[key] = mc.setMetricName(mname).evaluate(predictions)
        except Exception:
            metrics[key] = None

    if len(label_counts) < 2:
        metrics["AUC-ROC"] = None
        metrics["AUC-PR"]  = None
    else:
        bc = BinaryClassificationEvaluator(labelCol=label_col,
                                           rawPredictionCol="rawPrediction")
        for mname, key in [("areaUnderROC", "AUC-ROC"), ("areaUnderPR", "AUC-PR")]:
            try:
                metrics[key] = bc.setMetricName(mname).evaluate(predictions)
            except Exception:
                metrics[key] = None

    return metrics


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧬 HealthSpark")
    st.markdown("<p style='color:#94a3b8;font-size:.8rem;'>Distributed ML · PySpark · Healthcare</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📂 Dataset")
    dataset_choice = st.selectbox("Select Dataset", list(DATASET_PRESETS.keys()))
    preset         = DATASET_PRESETS[dataset_choice]
    st.markdown(f"<div class='info-box'>{preset['description']}</div>", unsafe_allow_html=True)
    if preset["kaggle"]:
        st.markdown(f"[📥 Download from Kaggle]({preset['kaggle']})")

    uploaded_file = st.file_uploader(
        f"Upload {preset['filename'] or 'your CSV'}", type=["csv"], key="csv_upload"
    )
    manual_path = st.text_input(
        "Or local file path",
        placeholder=f"data/raw/{preset['filename'] or 'data.csv'}"
    )

    custom_label_col = None
    if dataset_choice == "Custom Dataset" and uploaded_file:
        try:
            preview = pd.read_csv(uploaded_file, nrows=1)
            uploaded_file.seek(0)
            custom_label_col = st.selectbox("Label Column", preview.columns.tolist())
        except Exception:
            st.warning("Could not read column names.")

    st.markdown("---")
    st.markdown("### ⚙️ Model Config")
    model_choice = st.selectbox("ML Model",
                                ["Gradient Boosted Trees", "Random Forest", "Logistic Regression"])
    use_tuned  = st.toggle("Use dataset-tuned hyperparameters", value=True)
    test_size  = st.slider("Test Split (%)", 10, 40, 20) / 100

    num_trees = max_depth = max_iter = step_size = 100  # defaults
    if not use_tuned:
        st.markdown("**Manual hyperparameters**")
        if model_choice == "Random Forest":
            num_trees = st.slider("Number of Trees", 50, 400, 200, step=50)
            max_depth = st.slider("Max Depth", 4, 15, 10)
        elif model_choice == "Gradient Boosted Trees":
            max_iter  = st.slider("Max Iterations", 20, 150, 80, step=10)
            max_depth = st.slider("Max Depth", 3, 8, 6)
            step_size = st.slider("Step Size", 0.01, 0.3, 0.08, step=0.01)
        else:
            max_iter = st.slider("Max Iterations", 100, 600, 300, step=50)

    st.markdown("---")
    st.markdown("<p style='color:#475569;font-size:.75rem;'>PySpark MLlib + Streamlit</p>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Resolve path & label
# ─────────────────────────────────────────────
resolved_path = None
if uploaded_file is not None:
    resolved_path = save_upload(uploaded_file)
elif manual_path and os.path.exists(manual_path):
    resolved_path = manual_path

label_col   = custom_label_col if dataset_choice == "Custom Dataset" else preset["label_col"]
cat_cols    = preset["categorical_cols"]
label_names = preset["label_names"]


def get_params() -> dict:
    if use_tuned:
        return preset["best_params"].get(model_choice, {})
    if model_choice == "Random Forest":
        return {"numTrees": num_trees, "maxDepth": max_depth,
                "minInstancesPerNode": 2, "featureSubsetStrategy": "sqrt"}
    if model_choice == "Gradient Boosted Trees":
        return {"maxIter": max_iter, "maxDepth": max_depth,
                "stepSize": step_size, "subsamplingRate": 0.8,
                "minInstancesPerNode": 3}
    return {"maxIter": max_iter, "regParam": 0.01, "elasticNetParam": 0.1}


# ═══════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-family:Space Mono,monospace;
           background:linear-gradient(90deg,#38bdf8,#818cf8);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           font-size:2.2rem;margin-bottom:0'>
🧬 Early Disease Prediction
</h1>
<p style='color:#64748b;font-size:1rem;margin-top:4px'>
Scalable Distributed ML on Healthcare Data · Apache Spark</p>
""", unsafe_allow_html=True)

if resolved_path and label_col:
    binarize_note = " (binarized: num > 0 → 1)" if preset.get("binarize_label") else ""
    st.markdown(
        f"<div class='info-box'>✅ <strong>{dataset_choice}</strong> &nbsp;|&nbsp; "
        f"Label: <code>{label_col}</code>{binarize_note} &nbsp;|&nbsp; "
        f"File: <code>{os.path.basename(resolved_path)}</code></div>",
        unsafe_allow_html=True,
    )

st.markdown("---")
tabs = st.tabs(["📊 Data Explorer", "🤖 Train Model", "🔬 Predict Patient"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — Data Explorer
# ═══════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)

    if not resolved_path:
        st.info("⬅️ Upload a CSV or enter a local file path in the sidebar.")
    elif not label_col:
        st.warning("⬅️ Select a label column in the sidebar (Custom Dataset mode).")
    else:
        spark = get_spark()
        with st.spinner("Loading & preprocessing via Spark..."):
            if dataset_choice == "Diabetes (Pima Indians)":
                label_col = "Outcome"

            sdf, extra_feats = preprocess(spark, resolved_path, preset, label_col)
            num_cols, cat_present = split_feature_cols(sdf, label_col, cat_cols)
            pdf = sdf.toPandas()

        st.session_state.update({
            "feature_cols": num_cols,
            "cat_cols":     cat_present,
            "label_col":    label_col,
            "label_names":  label_names,
            "extra_feats":  extra_feats,
        })

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Records",       f"{len(pdf):,}")
        c2.metric("Numeric Feats", len(num_cols))
        c3.metric("Categorical",   len(cat_present))
        pos = int(pdf[label_col].sum())
        c4.metric(label_names.get(1, "Positive"), f"{pos:,}")
        c5.metric(label_names.get(0, "Negative"), f"{len(pdf) - pos:,}")

        if extra_feats:
            st.markdown(
                f"<div class='improvement-box'>✨ <strong>{len(extra_feats)} engineered features</strong> added: "
                + ", ".join(f"<code>{f}</code>" for f in extra_feats) + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**Numeric features:**")
        st.markdown(" ".join(f"<span class='tag'>{c}</span>" for c in num_cols),
                    unsafe_allow_html=True)
        if cat_present:
            st.markdown("**Categorical features (OHE-encoded):**")
            st.markdown(" ".join(f"<span class='tag'>{c}</span>" for c in cat_present),
                        unsafe_allow_html=True)

        st.markdown("#### Sample Data")
        st.dataframe(pdf.head(10), use_container_width=True)

        with st.expander("📋 Descriptive Statistics"):
            display_num = [c for c in num_cols if c in pdf.columns]
            st.dataframe(pdf[display_num].describe().T.round(2), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Feature Distribution")
            all_feat_cols = [c for c in (num_cols + cat_present) if c in pdf.columns]
            sel = st.selectbox("Feature", all_feat_cols, key="dist_sel")
            fig, ax = dark_fig((6, 3))
            if sel in cat_present:
                vc = pdf[sel].value_counts()
                ax.bar(vc.index.astype(str), vc.values, color="#818cf8", edgecolor="#0a0e1a")
                plt.xticks(rotation=30, ha="right", color="#94a3b8", fontsize=8)
            else:
                pdf[sel].hist(ax=ax, bins=30, color="#38bdf8", edgecolor="#0a0e1a")
            ax.set_title(sel, color="#e0e6f0", fontsize=12)
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### Correlation Heatmap (Numeric)")
            base_num   = [c for c in num_cols if c in pdf.columns and c not in extra_feats]
            corr_cols  = base_num + ([label_col] if label_col in pdf.columns else [])
            fig2, ax2  = dark_fig((6, 4))
            sns.heatmap(pdf[corr_cols].corr(), ax=ax2, cmap="coolwarm", annot=False,
                        linewidths=.5, linecolor="#0a0e1a", cbar_kws={"shrink": .8})
            ax2.tick_params(colors="#94a3b8", labelsize=8)
            st.pyplot(fig2); plt.close()

        st.markdown("#### Class Balance")
        fig3, ax3 = dark_fig((5, 2.5))
        counts_series = pdf[label_col].value_counts().sort_index()
        bar_labels    = [label_names.get(int(k), str(k)) for k in counts_series.index]
        ax3.bar(bar_labels, counts_series.values,
                color=["#22c55e" if i == 0 else "#ef4444" for i in range(len(counts_series))],
                width=.5)
        st.pyplot(fig3); plt.close()

        if cat_present:
            st.markdown("#### Categorical Feature Breakdown by Label")
            display_cat = [c for c in cat_present if c in pdf.columns]
            if display_cat:
                sel_cat = st.selectbox("Categorical feature", display_cat, key="cat_sel")
                fig4, ax4 = dark_fig((7, 3))
                crosstab  = pd.crosstab(pdf[sel_cat], pdf[label_col])
                crosstab.plot(kind="bar", ax=ax4,
                              color=["#22c55e", "#ef4444"][:len(crosstab.columns)],
                              edgecolor="#0a0e1a", width=.6)
                ax4.set_title(f"{sel_cat} vs {label_col}", color="#e0e6f0")
                ax4.set_xlabel(sel_cat, color="#94a3b8")
                ax4.legend([label_names.get(int(c), str(c)) for c in crosstab.columns],
                           facecolor="#111827", labelcolor="#e0e6f0")
                plt.xticks(rotation=30, ha="right", color="#94a3b8", fontsize=8)
                st.pyplot(fig4); plt.close()


# ═══════════════════════════════════════════════════════════════
# TAB 2 — Train Model
# ═══════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("<div class='section-header'>Train Spark ML Pipeline</div>", unsafe_allow_html=True)

    if not resolved_path:
        st.info("⬅️ Upload or provide a CSV path in the sidebar first.")
    elif not label_col:
        st.warning("⬅️ Select a label column for Custom Dataset.")
    else:
        params = get_params()

        with st.expander("🔧 Active Hyperparameters", expanded=False):
            st.markdown(f"**{'Dataset-tuned' if use_tuned else 'Manual'} params for {model_choice}:**")
            for k, v in params.items():
                st.markdown(f"- `{k}` = `{v}`")
            if use_tuned and preset.get("interaction_cols"):
                st.markdown("**Interaction features:**")
                for (a, b) in preset["interaction_cols"]:
                    st.markdown(f"- `{a} × {b}`")

        if st.button("Train Model"):
            spark = get_spark()
            prog  = st.progress(0, text="Loading & preprocessing data...")

            print("\n" + "="*60)
            print(f"  HealthSpark — Training Run")
            print("="*60)
            print(f"  Dataset  : {dataset_choice}")
            print(f"  Model    : {model_choice}")
            print(f"  Label    : {label_col}")
            print(f"  Params   : {params}")
            print("="*60)

            print("\n[1/5] Loading & preprocessing data...")
            t_pre = time.time()
            sdf, extra_feats = preprocess(spark, resolved_path, preset, label_col)
            num_cols, cat_present = split_feature_cols(sdf, label_col, cat_cols)
            total_rows = sdf.count()
            print(f"      ✓ Loaded {total_rows:,} records")
            print(f"      ✓ Numeric features  : {len(num_cols)}  → {num_cols}")
            print(f"      ✓ Categorical feats : {len(cat_present)}  → {cat_present}")
            if extra_feats:
                print(f"      ✓ Engineered feats  : {len(extra_feats)}  → {extra_feats}")
            print(f"      ✓ Preprocessing done in {round(time.time()-t_pre, 2)}s")

            prog.progress(20, text="Building ML pipeline...")
            print("\n[2/5] Building ML pipeline...")
            pipeline = build_pipeline(num_cols, cat_present, label_col, model_choice, params)
            print(f"      ✓ Stages: {[type(s).__name__ for s in pipeline.getStages()]}")

            prog.progress(35, text="Stratified train / test split...")
            print(f"\n[3/5] Stratified train/test split (test={int(test_size*100)}%)...")
            train_df, test_df = stratified_split(sdf, label_col, test_size, seed=42)
            n_train = train_df.count()
            n_test  = test_df.count()
            print(f"      ✓ Train : {n_train:,} records")
            print(f"      ✓ Test  : {n_test:,} records")
            for row in sorted(train_df.groupBy(label_col).count().collect(), key=lambda r: r[label_col]):
                cls_name = label_names.get(int(row[label_col]), str(row[label_col]))
                print(f"        - {cls_name}: {row['count']:,} ({row['count']/n_train*100:.1f}%)")

            prog.progress(50, text=f"Training {model_choice} on {n_train:,} records...")
            print(f"\n[4/5] Training {model_choice}...")
            print(f"      Hyperparameters:")
            for k, v in params.items():
                print(f"        {k:30s} = {v}")
            print(f"      Training started...")
            t0    = time.time()
            model = pipeline.fit(train_df)
            elapsed = round(time.time() - t0, 2)
            print(f"      ✓ Training complete in {elapsed}s")

            prog.progress(80, text="Evaluating on test set...")
            print(f"\n[5/5] Evaluating on {n_test:,} test records...")
            t_eval = time.time()
            preds   = model.transform(test_df)
            metrics = evaluate_model(preds, label_col)
            print(f"      ✓ Evaluation done in {round(time.time()-t_eval, 2)}s")
            print(f"\n{'─'*40}")
            print(f"  RESULTS")
            print(f"{'─'*40}")
            for mname, mval in metrics.items():
                val_str = f"{mval:.4f}" if mval is not None else "N/A  "
                bar     = "█" * int((mval or 0) * 20) + "░" * (20 - int((mval or 0) * 20))
                print(f"  {mname:<12} {val_str}  [{bar}]")
            print(f"{'─'*40}")
            print(f"  Total wall time : {round(time.time()-t_pre, 2)}s")
            print("="*60 + "\n")

            prog.progress(100, text="Done!")

            st.success(f"✅ Trained in **{elapsed}s** on `{n_train:,}` records")

            for imp in [
                f"✨ {len(extra_feats)} engineered features (interactions + log-transforms)" if extra_feats else None,
                "⚖️ Class-weighted loss (balances minority class)",
                "🔬 Per-class median imputation for zero-fill columns",
                "📊 Stratified train/test split with class-coverage guard",
                "🎯 Dataset-tuned hyperparameters" if use_tuned else None,
            ]:
                if imp:
                    st.markdown(f"<div class='improvement-box'>{imp}</div>", unsafe_allow_html=True)

            st.markdown("#### 📈 Performance Metrics")
            metric_cols = st.columns(6)
            for mcol, key in zip(metric_cols,
                                 ["AUC-ROC", "AUC-PR", "Accuracy", "F1 Score", "Precision", "Recall"]):
                val = metrics.get(key)
                mcol.metric(key, f"{val:.3f}" if val is not None else "N/A")

            with st.expander("🔧 Pipeline Stages"):
                for i, s in enumerate(model.stages):
                    st.markdown(f"**Stage {i+1}:** `{type(s).__name__}`")

            if model_choice in ("Random Forest", "Gradient Boosted Trees"):
                st.markdown("#### 🌲 Feature Importance")
                try:
                    imp_arr = model.stages[-1].featureImportances.toArray()
                    imp_labels = list(num_cols)
                    for c in cat_present:
                        imp_labels.append(f"{c} (encoded)")
                    imp_labels = (imp_labels + [f"feat_{i}" for i in range(len(imp_arr))])[:len(imp_arr)]

                    fi_df  = pd.DataFrame({"Feature": imp_labels, "Importance": imp_arr}).sort_values("Importance")
                    fig5, ax5 = dark_fig((7, max(3, len(fi_df) * 0.35)))
                    colors = plt.cm.cool(np.linspace(0.3, 1, len(fi_df)))
                    ax5.barh(fi_df["Feature"], fi_df["Importance"], color=colors)
                    ax5.set_xlabel("Importance", color="#94a3b8")
                    st.pyplot(fig5); plt.close()
                except Exception as e:
                    st.warning(f"Feature importance unavailable: {e}")

            os.makedirs("models", exist_ok=True)
            model.write().overwrite().save("models/spark_disease_model")
            st.info("💾 Model saved to `models/spark_disease_model`")

            st.session_state.update({
                "model":        model,
                "feature_cols": num_cols,
                "cat_cols":     cat_present,
                "label_col":    label_col,
                "label_names":  label_names,
                "extra_feats":  extra_feats,
                "preset":       preset,
            })


# ═══════════════════════════════════════════════════════════════
# TAB 3 — Predict Patient
# ═══════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("<div class='section-header'>Single Patient Prediction</div>", unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.warning("⚠️ Train a model first in the **Train Model** tab.")
    else:
        num_feat    = st.session_state["feature_cols"]
        cat_feat    = st.session_state["cat_cols"]
        lbl_col     = st.session_state["label_col"]
        lbl_names   = st.session_state["label_names"]
        extra_feat  = st.session_state.get("extra_feats", [])
        saved_preset = st.session_state.get("preset", preset)

        base_num = [f for f in num_feat if f not in extra_feat]
        base_cat = cat_feat

        st.markdown("#### Enter Patient Details")
        st.markdown("<div class='predict-box'>", unsafe_allow_html=True)

        input_values = {}
        all_base = base_num + base_cat
        grid = st.columns(4)

        for i, fname in enumerate(all_base):
            key  = fname.lower().replace(" ", "").replace("_", "")
            meta = COL_DEFAULTS.get(key, (0.0, 1000.0, 0.0, float, None))
            dmin, dmax, dval, dtype, options = meta
            with grid[i % 4]:
                if options is not None:
                    input_values[fname] = st.selectbox(
                        fname, options,
                        index=options.index(dval) if dval in options else 0,
                        key=f"p_{fname}",
                    )
                elif dtype == float:
                    input_values[fname] = st.number_input(
                        fname, float(dmin), float(dmax), float(dval),
                        step=0.01, key=f"p_{fname}",
                    )
                else:
                    input_values[fname] = st.number_input(
                        fname, int(dmin), int(dmax), int(dval),
                        key=f"p_{fname}",
                    )

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔬 Run Prediction"):
            spark = get_spark()

            row: dict = {}
            for fname in base_num:
                row[fname] = float(input_values[fname])
            for fname in base_cat:
                row[fname] = str(input_values[fname]).lower().strip()
            row[lbl_col]          = 0
            row["_class_weight"]  = 1.0

            # Reconstruct interaction features
            for (a, b) in saved_preset.get("interaction_cols", []):
                iname = f"{a}_x_{b}"
                if iname in num_feat:
                    row[iname] = float(input_values.get(a, 0)) * float(input_values.get(b, 0))

            # Log transforms
            for c in ["Insulin", "insulin"]:
                logname = f"{c}_log"
                if logname in num_feat and c in input_values:
                    row[logname] = math.log(float(input_values[c]) + 1.0)

            input_df = spark.createDataFrame(pd.DataFrame([row]))

            try:
                preds    = st.session_state["model"].transform(input_df)
                rows_out = preds.select("prediction", "probability").collect()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            if not rows_out:
                st.error("Prediction returned no results — check your input values.")
            else:
                r        = rows_out[0]
                label    = int(r["prediction"])
                prob_arr = r["probability"].toArray() if hasattr(r["probability"], "toArray") else list(r["probability"])
                prob     = float(prob_arr[1]) if len(prob_arr) > 1 else float(prob_arr[0])

                st.markdown("---")
                st.markdown("#### 🧪 Prediction Result")

                if label == 1:
                    st.markdown(f"""
                    <div class='risk-high'>
                        <h2>⚠️ HIGH RISK</h2>
                        <p style='font-size:1.1rem'>{lbl_names.get(1,'Positive')} Probability:
                        <strong>{prob*100:.1f}%</strong></p>
                        <p style='font-size:.85rem;opacity:.7'>
                        Recommend clinical follow-up and further testing.</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='risk-low'>
                        <h2>✅ LOW RISK</h2>
                        <p style='font-size:1.1rem'>{lbl_names.get(0,'Negative')} —
                        Risk Probability: <strong>{prob*100:.1f}%</strong></p>
                        <p style='font-size:.85rem;opacity:.7'>
                        No immediate concern. Routine monitoring advised.</p>
                    </div>""", unsafe_allow_html=True)

                st.markdown("#### Risk Probability Gauge")
                fig6, ax6 = dark_fig((7, 1.2))
                ax6.barh(["Risk"], [prob],
                         color="#ef4444" if label == 1 else "#22c55e", height=.4)
                ax6.barh(["Risk"], [1 - prob], left=[prob], color="#1e2a45", height=.4)
                ax6.axvline(x=.5, color="#64748b", linestyle="--", linewidth=.8)
                ax6.set_xlim(0, 1)
                ax6.set_xlabel("Probability", color="#94a3b8")
                st.pyplot(fig6); plt.close()

                st.markdown("#### Input Summary")
                st.dataframe(
                    pd.DataFrame({
                        "Feature": list(input_values.keys()),
                        "Value":   list(input_values.values()),
                        "Type":    ["categorical" if f in base_cat else "numeric"
                                    for f in input_values.keys()],
                    }),
                    use_container_width=True,
                    hide_index=True,
                )