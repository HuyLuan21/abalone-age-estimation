"""Data loading, validation, anomaly cleaning, and splitting module for Abalone dataset."""

import os
from typing import Dict, Tuple, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMNS = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
    "Rings",
]

NUMERIC_FEATURES = [
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
]

CATEGORICAL_FEATURES = ["Sex"]
TARGET_NAME = "Rings"

# Specific row-level anomaly definitions (verified against raw UCI dataset)
KNOWN_ANOMALIES_METADATA = {
    1210: "Inconsistent geometry: Diameter (0.375) > Length (0.185). Length is by definition the maximum dimension.",
    1257: "Biologically impossible: Height = 0.0 mm (shell height cannot be zero in living organism).",
    2051: "Data entry outlier typo: Height = 1.130 mm (~10x sample mean 0.140, exceeds shell Length 0.455).",
    3996: "Biologically impossible: Height = 0.0 mm (shell height cannot be zero in living organism).",
}


def load_data(data_path: str = "data/raw/abalone.data") -> pd.DataFrame:
    """Load raw dataset from CSV/data file with predefined column names.

    Args:
        data_path: Path to the raw abalone dataset.

    Returns:
        pd.DataFrame: Cleanly loaded DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please run 'python src/download_data.py' to acquire it."
        )

    df = pd.read_csv(data_path, header=None, names=COLUMNS)
    return df


def clean_anomalies(
    df: pd.DataFrame,
    audit_report_path: Optional[str] = "reports/removed_anomalies.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify and filter out physically impossible observations and extreme recording errors.

    Identifies exactly 4 observations:
    1. Row index 1210: Diameter > Length (geometric violation; shell measurement axis error).
    2. Row index 1257: Height == 0.0 (physical impossibility).
    3. Row index 2051: Height == 1.130 (extreme data entry typo, ~10x average height 0.14).
    4. Row index 3996: Height == 0.0 (physical impossibility).

    Args:
        df: Input raw DataFrame (4,177 samples).
        audit_report_path: Optional path to save the anomaly audit CSV.

    Returns:
        Tuple of (df_cleaned, df_anomalies).
    """
    cond_height_zero = df["Height"] <= 0
    cond_height_extreme = df["Height"] >= 1.0
    cond_diameter_gt_length = df["Diameter"] > df["Length"]

    anomaly_mask = cond_height_zero | cond_height_extreme | cond_diameter_gt_length
    df_anomalies = df[anomaly_mask].copy()

    # Annotate specific scientific rationale
    reasons = []
    for idx in df_anomalies.index:
        reasons.append(KNOWN_ANOMALIES_METADATA.get(idx, "Physical inconsistency or biological impossibility."))
    df_anomalies["Anomaly_Reason"] = reasons

    if audit_report_path:
        os.makedirs(os.path.dirname(audit_report_path), exist_ok=True)
        df_anomalies.to_csv(audit_report_path, index=True, index_label="Raw_Row_Index")

    df_cleaned = df[~anomaly_mask].copy().reset_index(drop=True)
    return df_cleaned, df_anomalies


def load_and_clean_data(
    data_path: str = "data/raw/abalone.data",
    audit_report_path: str = "reports/removed_anomalies.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw dataset and filter anomalies prior to any model processing or splitting."""
    raw_df = load_data(data_path)
    clean_df, anomalies_df = clean_anomalies(raw_df, audit_report_path=audit_report_path)
    return clean_df, anomalies_df


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform schema validation and summary checks on data quality.

    Args:
        df: Input DataFrame.

    Returns:
        Dict[str, Any]: Dictionary containing quality metrics and statistics.
    """
    missing_values = df.isnull().sum().to_dict()
    total_duplicates = int(df.duplicated().sum())
    sex_distribution = df["Sex"].value_counts().to_dict()
    rings_stats = df["Rings"].describe().to_dict()

    # Physical checks
    invalid_heights = int((df["Height"] <= 0).sum())
    extreme_heights = int((df["Height"] >= 1.0).sum())
    diameter_greater_length = int((df["Diameter"] > df["Length"]).sum())

    report = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": missing_values,
        "duplicate_rows": total_duplicates,
        "sex_distribution": sex_distribution,
        "rings_stats": rings_stats,
        "invalid_heights": invalid_heights,
        "extreme_heights": extreme_heights,
        "diameter_greater_length": diameter_greater_length,
    }
    return report


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    auto_clean: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into Train and Test sets before any pre-processing or modeling.

    Ensures strict separation to prevent data leakage. If raw data with anomalies is provided,
    cleaning is strictly executed BEFORE splitting.

    Args:
        df: Full DataFrame containing features and target.
        test_size: Ratio of the test set (default 0.2 for 80/20 split).
        random_state: Random seed for reproducibility.
        auto_clean: Whether to ensure anomalies are filtered before split if present.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if auto_clean:
        # Check if df contains known anomalies and clean before splitting
        has_anomalies = (df["Height"] <= 0).any() or (df["Height"] >= 1.0).any() or (df["Diameter"] > df["Length"]).any()
        if has_anomalies:
            df, _ = clean_anomalies(df)

    X = df.drop(columns=[TARGET_NAME])
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("=== Testing data.py ===")
    dataset = load_data()
    quality = check_data_quality(dataset)
    print(f"Loaded raw dataset: {quality['num_rows']} rows, {quality['num_columns']} columns")
    clean_df, anomalies_df = clean_anomalies(dataset)
    print(f"Cleaned dataset: {len(clean_df)} rows, Filtered anomalies: {len(anomalies_df)} rows")
    X_tr, X_te, y_tr, y_te = split_data(clean_df)
    print(f"Train set: {X_tr.shape}, Test set: {X_te.shape}")
