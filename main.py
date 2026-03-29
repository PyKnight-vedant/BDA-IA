import os
import sys

# ── CRITICAL: Set Python executable env vars BEFORE any Spark imports
# This fixes "Python worker failed to connect back" on Windows
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import time
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean as spark_mean, trim, lower
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthSpark — Early Disease Prediction",
    page_icon="🧬", layout="wide", initial_sidebar_state="expanded"
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
    border-radius:6px; padding:2px 10px; font-size:.78rem; color:#94a3b8;
    margin:2px;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Spark — env vars already set at module level above
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_spark():
    spark = SparkSession.builder \
        .appName("HealthSpark") \
        .master("local[*]") \
        .config("spark.driver.memory",             "4g") \
        .config("spark.sql.shuffle.partitions",    "8") \
        .config("spark.pyspark.python",            sys.executable) \
        .config("spark.pyspark.driver.python",     sys.executable) \
        .config("spark.python.worker.reuse",       "true") \
        .config("spark.driver.host",               "127.0.0.1") \
        .config("spark.driver.bindAddress",        "127.0.0.1") \
        .config("spark.python.worker.timeout",     "120") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# ──────────────────────────────────────────────────────────────
# Dataset Presets
# ──────────────────────────────────────────────────────────────
DATASET_PRESETS = {
    "Cleveland Heart Disease (UCI Full)": {
        "description": "Full UCI Heart Disease dataset with categorical features & multi-site origin",
        "label_col": "num",
        "label_names": {0: "No Disease", 1: "Heart Disease"},
        "binarize_label": True,
        "drop_cols": ["id", "origin"],
        "categorical_cols": ["sex", "cp", "restecg", "exang", "thal", "slope"],
        "zero_fill_cols": [],
        "kaggle": "https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data",
        "filename": "heart_disease_uci.csv",
    },
    "Diabetes (Pima Indians)": {
        "description": "Predict diabetes onset in female Pima Indian patients",
        "label_col": "Outcome",
        "label_names": {0: "Non-Diabetic", 1: "Diabetic"},
        "binarize_label": False,
        "drop_cols": [],
        "categorical_cols": [],
        "zero_fill_cols": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "kaggle": "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database",
        "filename": "diabetes.csv",
    },
    "Custom Dataset": {
        "description": "Upload any binary classification healthcare CSV",
        "label_col": None,
        "label_names": {0: "Negative", 1: "Positive"},
        "binarize_label": False,
        "drop_cols": [],
        "categorical_cols": [],
        "zero_fill_cols": [],
        "kaggle": None,
        "filename": None,
    },
}

# ──────────────────────────────────────────────────────────────
# Column meta for Predict tab (min, max, default, type, options)
# ──────────────────────────────────────────────────────────────
COL_DEFAULTS = {
    "pregnancies":              (0,   20,    2,    int,   None),
    "glucose":                  (50,  300,   120,  int,   None),
    "bloodpressure":            (30,  140,   72,   int,   None),
    "skinthickness":            (0,   100,   23,   int,   None),
    "insulin":                  (0,   900,   80,   int,   None),
    "bmi":                      (10,  70,    28.5, float, None),
    "diabetespedigreefunction": (0,   3,     0.45, float, None),
    "age":                      (1,   120,   35,   int,   None),
    "trestbps":                 (80,  200,   130,  int,   None),
    "chol":                     (100, 600,   246,  int,   None),
    "thalach":                  (60,  220,   150,  int,   None),
    "oldpeak":                  (0,   10,    1.0,  float, None),
    "ca":                       (0,   3,     0,    int,   None),
    "fbs":                      (0,   1,     0,    int,   None),
    "sex":     (None, None, "Male",             int,   ["Male", "Female"]),
    "cp":      (None, None, "typical angina",   str,   ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]),
    "restecg": (None, None, "normal",           str,   ["normal", "stt abnormality", "lv hypertrophy"]),
    "exang":   (None, None, "FALSE",            str,   ["TRUE", "FALSE"]),
    "thal":    (None, None, "normal",           str,   ["normal", "fixed defect", "reversible defect"]),
    "slope":   (None, None, "upsloping",        str,   ["upsloping", "flat", "downsloping"]),
}

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def save_upload(f):
    suffix = os.path.splitext(f.name)[-1] or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(f.getvalue()); tmp.flush(); tmp.close()
    return tmp.name

def load_csv(spark, path):
    return spark.read.csv(path, header=True, inferSchema=True)

def preprocess(spark, path, preset, label_col):
    """Full preprocessing: load → drop → binarize → clean zeros → fill nulls."""
    sdf = load_csv(spark, path)

    cat_cols = preset.get("categorical_cols", [])
    for c in cat_cols:
        if c in sdf.columns:
            sdf = sdf.withColumn(c, lower(trim(col(c).cast("string"))))

    for c in preset.get("drop_cols", []):
        if c in sdf.columns:
            sdf = sdf.drop(c)

    if preset.get("binarize_label") and label_col in sdf.columns:
        sdf = sdf.withColumn(label_col,
              when(col(label_col) > 0, 1).otherwise(0).cast("int"))

    for c in preset.get("zero_fill_cols", []):
        if c in sdf.columns:
            sdf = sdf.withColumn(c, when(col(c) == 0, None).otherwise(col(c)))

    for f in sdf.schema.fields:
        if f.name == label_col or f.name in cat_cols:
            continue
        if any(t in str(f.dataType).lower() for t in ("int","double","float","decimal","long")):
            try:
                mv = sdf.select(spark_mean(col(f.name))).collect()[0][0]
                if mv is not None:
                    sdf = sdf.fillna({f.name: round(float(mv), 4)})
            except Exception:
                pass

    for c in cat_cols:
        if c in sdf.columns:
            try:
                mode_row = sdf.groupBy(c).count().orderBy("count", ascending=False).first()
                if mode_row:
                    sdf = sdf.fillna({c: mode_row[0]})
            except Exception:
                pass

    return sdf

def split_feature_cols(sdf, label_col, cat_cols):
    numeric_types = {"int","bigint","double","float","decimal","long"}
    num_cols = [
        f.name for f in sdf.schema.fields
        if f.name != label_col
        and f.name not in cat_cols
        and any(t in str(f.dataType).lower() for t in numeric_types)
    ]
    cat_present = [c for c in cat_cols if c in sdf.columns]
    return num_cols, cat_present

def build_pipeline(num_cols, cat_cols, label_col, model_name, max_iter=100, num_trees=100):
    stages = []
    final_feature_cols = list(num_cols)

    for c in cat_cols:
        idx_out = f"{c}_idx"
        ohe_out = f"{c}_ohe"
        stages.append(StringIndexer(inputCol=c, outputCol=idx_out,
                                    handleInvalid="keep", stringOrderType="frequencyDesc"))
        stages.append(OneHotEncoder(inputCols=[idx_out], outputCols=[ohe_out], handleInvalid="keep"))
        final_feature_cols.append(ohe_out)

    stages.append(VectorAssembler(inputCols=final_feature_cols, outputCol="raw_features",
                                  handleInvalid="skip"))
    stages.append(StandardScaler(inputCol="raw_features", outputCol="features",
                                 withStd=True, withMean=True))

    if model_name == "Random Forest":
        clf = RandomForestClassifier(labelCol=label_col, featuresCol="features",
                                     numTrees=num_trees, seed=42)
    elif model_name == "Gradient Boosted Trees":
        clf = GBTClassifier(labelCol=label_col, featuresCol="features",
                            maxIter=max_iter, seed=42)
    else:
        clf = LogisticRegression(labelCol=label_col, featuresCol="features",
                                 maxIter=max_iter)
    stages.append(clf)
    return Pipeline(stages=stages), final_feature_cols

def evaluate_model(predictions, label_col):
    bin_eval = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC")
    mc_eval  = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction")
    return {
        "AUC-ROC":   bin_eval.evaluate(predictions),
        "Accuracy":  mc_eval.setMetricName("accuracy").evaluate(predictions),
        "F1 Score":  mc_eval.setMetricName("f1").evaluate(predictions),
        "Precision": mc_eval.setMetricName("weightedPrecision").evaluate(predictions),
        "Recall":    mc_eval.setMetricName("weightedRecall").evaluate(predictions),
    }

def dark_fig(figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#111827'); ax.set_facecolor('#111827')
    ax.tick_params(colors='#94a3b8')
    for s in ax.spines.values(): s.set_edgecolor('#2a3a5c')
    return fig, ax

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 HealthSpark")
    st.markdown("<p style='color:#94a3b8;font-size:.8rem;'>Distributed ML · PySpark · Healthcare</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📂 Dataset")
    dataset_choice = st.selectbox("Select Dataset", list(DATASET_PRESETS.keys()))
    preset = DATASET_PRESETS[dataset_choice]
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
            prev = pd.read_csv(uploaded_file, nrows=1); uploaded_file.seek(0)
            custom_label_col = st.selectbox("Label Column", prev.columns.tolist())
        except Exception:
            st.warning("Could not read column names.")

    st.markdown("---")
    st.markdown("### ⚙️ Model Config")
    model_choice = st.selectbox("ML Model",
                                ["Random Forest", "Gradient Boosted Trees", "Logistic Regression"])
    test_size = st.slider("Test Split (%)", 10, 40, 20) / 100
    if model_choice == "Random Forest":
        num_trees = st.slider("Number of Trees", 20, 200, 100, step=10)
        max_iter  = 100
    elif model_choice == "Gradient Boosted Trees":
        max_iter  = st.slider("Max Iterations", 10, 100, 50, step=10)
        num_trees = 100
    else:
        max_iter  = st.slider("Max Iterations", 50, 300, 100, step=50)
        num_trees = 100

    st.markdown("---")
    st.markdown("<p style='color:#475569;font-size:.75rem;'>PySpark MLlib + Streamlit</p>",
                unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Resolve path / label
# ──────────────────────────────────────────────────────────────
resolved_path = None
if uploaded_file is not None:
    resolved_path = save_upload(uploaded_file)
elif manual_path and os.path.exists(manual_path):
    resolved_path = manual_path

label_col      = custom_label_col if dataset_choice == "Custom Dataset" else preset["label_col"]
zero_fill_cols = preset["zero_fill_cols"]
cat_cols       = preset["categorical_cols"]
label_names    = preset["label_names"]

# ──────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Space Mono,monospace;background:linear-gradient(90deg,#38bdf8,#818cf8);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.2rem;margin-bottom:0'>
🧬 Early Disease Prediction
</h1>
<p style='color:#64748b;font-size:1rem;margin-top:4px'>
Scalable Distributed ML on Healthcare Data · Apache Spark</p>
""", unsafe_allow_html=True)

if resolved_path and label_col:
    st.markdown(
        f"<div class='info-box'>✅ <strong>{dataset_choice}</strong> &nbsp;|&nbsp; "
        f"Label: <code>{label_col}</code>"
        + (f" (binarized: num > 0 → 1)" if preset.get("binarize_label") else "") +
        f" &nbsp;|&nbsp; File: <code>{os.path.basename(resolved_path)}</code></div>",
        unsafe_allow_html=True
    )

st.markdown("---")
tabs = st.tabs(["📊 Data Explorer", "🤖 Train Model", "🔬 Predict Patient"])

# ──────────────────────────────────────────────────────────────
# TAB 1 — Data Explorer
# ──────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)

    if not resolved_path:
        st.info("⬅️ Upload a CSV or enter a local file path in the sidebar.")
    elif not label_col:
        st.warning("⬅️ Select a label column in the sidebar (Custom Dataset mode).")
    else:
        spark = get_spark()
        with st.spinner("Loading & preprocessing via Spark..."):
            sdf = preprocess(spark, resolved_path, preset, label_col)
            num_cols, cat_present = split_feature_cols(sdf, label_col, cat_cols)
            pdf = sdf.toPandas()   # toPandas() is safe here — no ML transform involved

        st.session_state.update({
            "feature_cols": num_cols,
            "cat_cols":     cat_present,
            "label_col":    label_col,
            "label_names":  label_names,
        })

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Records",        f"{len(pdf):,}")
        c2.metric("Numeric Feats",  len(num_cols))
        c3.metric("Categorical",    len(cat_present))
        pos = int(pdf[label_col].sum())
        c4.metric(label_names.get(1, "Positive"), f"{pos:,}")
        c5.metric(label_names.get(0, "Negative"), f"{len(pdf)-pos:,}")

        st.markdown("**Numeric features:**")
        st.markdown(" ".join(f"<span class='tag'>{c}</span>" for c in num_cols),
                    unsafe_allow_html=True)
        if cat_present:
            st.markdown("**Categorical features (will be OHE-encoded):**")
            st.markdown(" ".join(f"<span class='tag'>{c}</span>" for c in cat_present),
                        unsafe_allow_html=True)

        st.markdown("#### Sample Data")
        st.dataframe(pdf.head(10), use_container_width=True)

        with st.expander("📋 Descriptive Statistics"):
            st.dataframe(pdf[num_cols].describe().T.round(2), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Feature Distribution")
            all_feat_cols = num_cols + cat_present
            sel = st.selectbox("Feature", all_feat_cols, key="dist_sel")
            fig, ax = dark_fig((6, 3))
            if sel in cat_present:
                vc = pdf[sel].value_counts()
                ax.bar(vc.index.astype(str), vc.values, color='#818cf8', edgecolor='#0a0e1a')
                ax.set_title(sel, color='#e0e6f0', fontsize=12)
                plt.xticks(rotation=30, ha='right', color='#94a3b8', fontsize=8)
            else:
                pdf[sel].hist(ax=ax, bins=30, color='#38bdf8', edgecolor='#0a0e1a')
                ax.set_title(sel, color='#e0e6f0', fontsize=12)
            st.pyplot(fig); plt.close()

        with col2:
            st.markdown("#### Correlation Heatmap (Numeric)")
            corr_cols = num_cols + ([label_col] if label_col in pdf.columns else [])
            fig2, ax2 = dark_fig((6, 4))
            sns.heatmap(pdf[corr_cols].corr(), ax=ax2, cmap='coolwarm', annot=False,
                        linewidths=.5, linecolor='#0a0e1a', cbar_kws={'shrink': .8})
            ax2.tick_params(colors='#94a3b8', labelsize=8)
            st.pyplot(fig2); plt.close()

        st.markdown("#### Class Balance")
        fig3, ax3 = dark_fig((5, 2.5))
        counts = pdf[label_col].value_counts().sort_index()
        bar_labels = [label_names.get(int(k), str(k)) for k in counts.index]
        ax3.bar(bar_labels, counts.values,
                color=['#22c55e' if i == 0 else '#ef4444' for i in range(len(counts))],
                width=.5)
        st.pyplot(fig3); plt.close()

        if cat_present:
            st.markdown("#### Categorical Feature Breakdown by Label")
            sel_cat = st.selectbox("Categorical feature", cat_present, key="cat_sel")
            fig4, ax4 = dark_fig((7, 3))
            crosstab = pd.crosstab(pdf[sel_cat], pdf[label_col])
            crosstab.plot(kind='bar', ax=ax4,
                          color=['#22c55e', '#ef4444'][:len(crosstab.columns)],
                          edgecolor='#0a0e1a', width=.6)
            ax4.set_title(f"{sel_cat} vs {label_col}", color='#e0e6f0')
            ax4.set_xlabel(sel_cat, color='#94a3b8')
            ax4.legend([label_names.get(int(c), str(c)) for c in crosstab.columns],
                       facecolor='#111827', labelcolor='#e0e6f0')
            plt.xticks(rotation=30, ha='right', color='#94a3b8', fontsize=8)
            st.pyplot(fig4); plt.close()

# ──────────────────────────────────────────────────────────────
# TAB 2 — Train Model
# ──────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("<div class='section-header'>Train Spark ML Pipeline</div>", unsafe_allow_html=True)

    if not resolved_path:
        st.info("⬅️ Upload or provide a CSV path in the sidebar first.")
    elif not label_col:
        st.warning("⬅️ Select a label column for Custom Dataset.")
    else:
        if st.button("🚀 Train Model"):
            spark = get_spark()
            prog  = st.progress(0, text="Loading & preprocessing data...")

            sdf = preprocess(spark, resolved_path, preset, label_col)
            num_cols, cat_present = split_feature_cols(sdf, label_col, cat_cols)

            prog.progress(20, text="Building Spark ML pipeline (StringIndexer → OHE → Scaler → Model)...")
            pipeline, final_feat_cols = build_pipeline(
                num_cols, cat_present, label_col, model_choice, max_iter, num_trees
            )

            prog.progress(35, text="Splitting train / test...")
            train_df, test_df = sdf.randomSplit([1 - test_size, test_size], seed=42)
            n_train = train_df.count()

            prog.progress(50, text=f"Training {model_choice} on {n_train:,} records...")
            t0      = time.time()
            model   = pipeline.fit(train_df)
            elapsed = round(time.time() - t0, 2)

            prog.progress(80, text="Evaluating on test set...")
            preds   = model.transform(test_df)
            metrics = evaluate_model(preds, label_col)
            prog.progress(100, text="Done!")

            st.success(f"✅ Trained in **{elapsed}s** on `{n_train:,}` records")

            st.markdown("#### 📈 Performance Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("AUC-ROC",   f"{metrics['AUC-ROC']:.3f}")
            m2.metric("Accuracy",  f"{metrics['Accuracy']:.3f}")
            m3.metric("F1 Score",  f"{metrics['F1 Score']:.3f}")
            m4.metric("Precision", f"{metrics['Precision']:.3f}")
            m5.metric("Recall",    f"{metrics['Recall']:.3f}")

            with st.expander("🔧 Pipeline Stages"):
                for i, s in enumerate(model.stages):
                    st.markdown(f"**Stage {i+1}:** `{type(s).__name__}`")

            if model_choice == "Random Forest":
                st.markdown("#### 🌲 Feature Importance")
                imp = model.stages[-1].featureImportances.toArray()

                imp_labels = list(num_cols)
                for c in cat_present:
                    imp_labels.append(f"{c} (encoded)")
                while len(imp_labels) < len(imp):
                    imp_labels.append(f"feat_{len(imp_labels)}")
                imp_labels = imp_labels[:len(imp)]

                fi_df = pd.DataFrame({"Feature": imp_labels, "Importance": imp}) \
                          .sort_values("Importance", ascending=True)

                fig5, ax5 = dark_fig((7, max(3, len(fi_df) * 0.38)))
                colors = plt.cm.cool(np.linspace(0.3, 1, len(fi_df)))
                ax5.barh(fi_df["Feature"], fi_df["Importance"], color=colors)
                ax5.set_xlabel("Importance", color='#94a3b8')
                st.pyplot(fig5); plt.close()

            os.makedirs("models", exist_ok=True)
            model.write().overwrite().save("models/spark_disease_model")
            st.info("💾 Model saved to `models/spark_disease_model`")

            st.session_state.update({
                "model":        model,
                "feature_cols": num_cols,
                "cat_cols":     cat_present,
                "label_col":    label_col,
                "label_names":  label_names,
            })

# ──────────────────────────────────────────────────────────────
# TAB 3 — Predict Patient
# ──────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("<div class='section-header'>Single Patient Prediction</div>", unsafe_allow_html=True)

    if "model" not in st.session_state:
        st.warning("⚠️ Train a model first in the **Train Model** tab.")
    else:
        num_feat  = st.session_state["feature_cols"]
        cat_feat  = st.session_state["cat_cols"]
        lbl_col   = st.session_state["label_col"]
        lbl_names = st.session_state["label_names"]

        st.markdown("#### Enter Patient Details")
        st.markdown("<div class='predict-box'>", unsafe_allow_html=True)

        all_feats    = num_feat + cat_feat
        input_values = {}
        grid = st.columns(4)

        for i, fname in enumerate(all_feats):
            key  = fname.lower().replace(" ", "").replace("_", "")
            meta = COL_DEFAULTS.get(key, (0.0, 1000.0, 0.0, float, None))
            dmin, dmax, dval, dtype, options = meta

            with grid[i % 4]:
                if options is not None:
                    input_values[fname] = st.selectbox(fname, options,
                                                       index=options.index(dval) if dval in options else 0,
                                                       key=f"p_{fname}")
                elif dtype == float:
                    input_values[fname] = st.number_input(
                        fname, float(dmin), float(dmax), float(dval), step=0.01, key=f"p_{fname}"
                    )
                else:
                    input_values[fname] = st.number_input(
                        fname, int(dmin), int(dmax), int(dval), key=f"p_{fname}"
                    )

        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔬 Run Prediction"):
            spark = get_spark()

            row = {}
            for fname in num_feat:
                row[fname] = float(input_values[fname])
            for fname in cat_feat:
                row[fname] = str(input_values[fname]).lower().strip()
            row[lbl_col] = 0.0  # dummy label

            input_df = spark.createDataFrame(pd.DataFrame([row]))
            preds    = st.session_state["model"].transform(input_df)

            # ── FIX: use .collect() instead of .toPandas() after an ML transform.
            # On Windows, .toPandas() on a transformed DataFrame tries to spawn a
            # fresh Python RDD worker via a socket handshake — which times out
            # ("Python worker failed to connect back").  .collect() returns the
            # data through the JVM↔Py4J gateway instead, bypassing that worker.
            try:
                rows = preds.select("prediction", "probability").collect()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

            if not rows:
                st.error("Prediction returned no results — check your input values.")
            else:
                row_result = rows[0]
                label      = int(row_result["prediction"])
                prob_vec   = row_result["probability"]
                prob_arr   = prob_vec.toArray() if hasattr(prob_vec, "toArray") else list(prob_vec)
                prob       = float(prob_arr[1]) if len(prob_arr) > 1 else float(prob_arr[0])

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
                ax6.barh(["Risk"], [prob],       color='#ef4444' if label == 1 else '#22c55e', height=.4)
                ax6.barh(["Risk"], [1 - prob],   left=[prob], color='#1e2a45', height=.4)
                ax6.axvline(x=.5, color='#64748b', linestyle='--', linewidth=.8)
                ax6.set_xlim(0, 1)
                ax6.set_xlabel("Probability", color='#94a3b8')
                st.pyplot(fig6); plt.close()

                st.markdown("#### Input Summary")
                st.dataframe(
                    pd.DataFrame({
                        "Feature": list(input_values.keys()),
                        "Value":   list(input_values.values()),
                        "Type":    ["categorical" if f in cat_feat else "numeric"
                                    for f in input_values.keys()]
                    }),
                    use_container_width=True, hide_index=True
                )