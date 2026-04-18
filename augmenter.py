import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

from pyspark.sql import SparkSession

FILE_PATH = r"D:\Vedant's Stuff\BDA\IA\diabetes copy.csv"
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# -------------------------------
# Local SMOTE (runs per partition)
# -------------------------------
def local_smote_partition(iterator):
    df_chunk = pd.DataFrame(iterator)

    if df_chunk.empty:
        return iter([])

    df_chunk.columns = global_columns

    X = df_chunk.drop(columns=[target_col]).values
    y = df_chunk[target_col].values

    minority = 1 if np.sum(y == 1) < np.sum(y == 0) else 0
    X_min = X[y == minority]

    if len(X_min) < 2:
        return df_chunk.to_dict("records")

    k = min(5, len(X_min) - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(X_min)

    synthetic = []

    for i in range(len(X_min)):
        xi = X_min[i]
        neigh_idx = nn.kneighbors([xi], return_distance=False)[0]

        for j in neigh_idx:
            xj = X_min[j]
            lam = np.random.rand()
            synthetic.append(xi + lam * (xj - xi))

    X_syn = np.array(synthetic)
    y_syn = np.full(len(X_syn), minority)

    X_all = np.vstack([X, X_syn])
    y_all = np.hstack([y, y_syn])

    df_out = pd.DataFrame(X_all, columns=df_chunk.columns[:-1])
    df_out[target_col] = y_all

    # noise
    df_out[df_chunk.columns[:-1]] += np.random.normal(
        0, noise_level, df_out[df_chunk.columns[:-1]].shape
    )

    return df_out.to_dict("records")


# -------------------------------
# MAIN
# -------------------------------
def augment_spark():
    spark = SparkSession.builder \
        .appName("Distributed_SMOTE") \
        .getOrCreate()

    df = pd.read_csv(FILE_PATH)
    df.columns = df.columns.str.strip()

    global global_columns, target_col, noise_level
    noise_level = 0.03

    # detect target
    target_col = next(
        (c for c in ["target", "num", "label", "y", "Outcome"] if c in df.columns),
        None
    )

    if target_col is None:
        raise ValueError(f"No target column found: {df.columns}")

    print("Using target:", target_col)

    # binary
    df[target_col] = (df[target_col] > 0).astype(int)

    # numeric only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(target_col)

    df = df[numeric_cols + [target_col]]

    # impute
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    global_columns = df.columns.tolist()

    # -------------------------------
    # Convert to Spark RDD
    # -------------------------------
    rdd = spark.sparkContext.parallelize(df.to_dict("records"))

    # -------------------------------
    # MAP (parallel SMOTE)
    # -------------------------------
    augmented_rdd = rdd.mapPartitions(local_smote_partition)

    # -------------------------------
    # REDUCE (implicit)
    # -------------------------------
    result = augmented_rdd.collect()

    df_res = pd.DataFrame(result)

    # expand
    df_res = pd.concat([df_res] * 3, ignore_index=True)

    # shuffle
    df_res = df_res.sample(frac=1).reset_index(drop=True)

    # save
    df_res.to_csv(FILE_PATH, index=False)

    print("Done. Final size:", len(df_res))


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    augment_spark()