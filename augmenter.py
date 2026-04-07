import pandas as pd
import numpy as np

def augment_csv(file_path, factor=100, noise_level=0.02):
    df = pd.read_csv(file_path)
    
    n = len(df)
    new_rows = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    for _ in range(factor * n):
        row = df.sample(1).iloc[0].copy()

        # Add noise to numeric columns
        for col in numeric_cols:
            std = df[col].std()
            if std > 0:
                noise = np.random.normal(0, noise_level * std)
                row[col] += noise

        # For categorical → random resample (keeps distribution)
        for col in non_numeric_cols:
            row[col] = df[col].sample(1).values[0]

        new_rows.append(row)

    df_aug = pd.DataFrame(new_rows)

    # Combine original + augmented
    df_final = pd.concat([df, df_aug], ignore_index=True)

    # Shuffle
    df_final = df_final.sample(frac=1).reset_index(drop=True)

    # Overwrite file
    df_final.to_csv(file_path, index=False)


# Usage
augment_csv(r"D:\Vedant's Stuff\BDA\IA\data\raw\diabetes.csv", factor=20, noise_level=0.03)