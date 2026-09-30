"""Data loading, validation, and splitting module for Abalone dataset."""

import os
from typing import Dict, Tuple, Any
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


def load_data(data_path: str = "data/raw/abalone.data") -> pd.DataFrame:
    """Load raw dataset from CSV/data file with predefined column names.

    Args:
        data_path: Path to the raw abalone dataset.

    Returns:
        pd.DataFrame: Cleanly loaded DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. Please check directory structure."
        )

    df = pd.read_csv(data_path, header=None, names=COLUMNS)
    return df


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

    # Domain check: physical measurements should be strictly positive
    invalid_heights = int((df["Height"] <= 0).sum())

    report = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "missing_values": missing_values,
        "duplicate_rows": total_duplicates,
        "sex_distribution": sex_distribution,
        "rings_stats": rings_stats,
        "invalid_heights": invalid_heights,
    }
    return report


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into Train and Test sets before any pre-processing or modeling.

    Ensures strict separation to prevent data leakage.

    Args:
        df: Full DataFrame containing features and target.
        test_size: Ratio of the test set (default 0.2 for 80/20 split).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
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
    print(f"Loaded dataset: {quality['num_rows']} rows, {quality['num_columns']} columns")
    print(f"Missing values: {quality['missing_values']}")
    print(f"Duplicates: {quality['duplicate_rows']}")
    X_tr, X_te, y_tr, y_te = split_data(dataset)
    print(f"Train set: {X_tr.shape}, Test set: {X_te.shape}")
