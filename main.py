# =====================================================
# PROJECT 06 - Abalone Age Estimation using K-NN Regression
# Theo yêu cầu môn Học máy - Bài 2
# =====================================================

import os
import math
# pyrefly: ignore [missing-import]
import joblib
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# Tạo thư mục lưu kết quả
# =====================================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# =====================================================
# BƯỚC 1 - Đọc dữ liệu
# =====================================================

columns = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "Whole_weight",
    "Shucked_weight",
    "Viscera_weight",
    "Shell_weight",
    "Rings"
]

df = pd.read_csv(
    "data/raw/abalone.data",
    header=None,
    names=columns
)

print("=" * 60)
print("THÔNG TIN DỮ LIỆU")
print("=" * 60)

print("Kích thước:", df.shape)
print(df.head())

print("\nKiểu dữ liệu:")
df.info()

# =====================================================
# BƯỚC 2 - Khám phá dữ liệu (EDA)
# =====================================================

print("\nGiá trị thiếu:")
print(df.isnull().sum())

print("\nPhân bố giới tính:")
print(df["Sex"].value_counts())

plt.figure(figsize=(8,5))
plt.hist(df["Rings"], bins=20)
plt.title("Distribution of Rings")
plt.xlabel("Rings")
plt.ylabel("Count")

plt.savefig(
    "reports/figures/rings_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# =====================================================
# BƯỚC 3 - Tiền xử lý
# =====================================================

df = pd.get_dummies(df, columns=["Sex"], dtype=int)

X = df.drop(columns=["Rings"])
y = df["Rings"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain:", X_train.shape)
print("Test :", X_test.shape)

# =====================================================
# BƯỚC 4 - Baseline: Linear Regression
# =====================================================

baseline = LinearRegression()
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = math.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)

print("\nBaseline - Linear Regression")
print(f"MAE : {baseline_mae:.3f}")
print(f"RMSE: {baseline_rmse:.3f}")
print(f"R²  : {baseline_r2:.3f}")

# =====================================================
# BƯỚC 5 - KNN KHÔNG SCALE
# =====================================================

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)

knn_mae = mean_absolute_error(y_test, pred)
knn_rmse = math.sqrt(mean_squared_error(y_test, pred))
knn_r2 = r2_score(y_test, pred)

print("\nKNN (Không Scale)")
print(f"MAE : {knn_mae:.3f}")
print(f"RMSE: {knn_rmse:.3f}")
print(f"R²  : {knn_r2:.3f}")

# =====================================================
# BƯỚC 6 - KNN CÓ STANDARD SCALER
# =====================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=5))
])

pipeline.fit(X_train, y_train)

scaled_pred = pipeline.predict(X_test)

scaled_mae = mean_absolute_error(y_test, scaled_pred)
scaled_rmse = math.sqrt(mean_squared_error(y_test, scaled_pred))
scaled_r2 = r2_score(y_test, scaled_pred)

print("\nKNN (Có StandardScaler)")
print(f"MAE : {scaled_mae:.3f}")
print(f"RMSE: {scaled_rmse:.3f}")
print(f"R²  : {scaled_r2:.3f}")

# =====================================================
# BƯỚC 7 - MAE theo K
# =====================================================

k_values = range(1,31)
mae_values = []

for k in k_values:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(n_neighbors=k))
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    mae_values.append(mae)

best_k = k_values[mae_values.index(min(mae_values))]

print(f"\nBest K = {best_k}")

plt.figure(figsize=(8,5))
plt.plot(k_values, mae_values, marker="o")
plt.xticks(k_values)
plt.grid(True)

plt.title("MAE theo K")
plt.xlabel("K")
plt.ylabel("MAE")

plt.savefig(
    "reports/figures/mae_vs_k.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# =====================================================
# BƯỚC 8 - Huấn luyện lại với Best K
# =====================================================

best_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsRegressor(n_neighbors=best_k))
])

best_pipeline.fit(X_train, y_train)

best_pred = best_pipeline.predict(X_test)

best_mae = mean_absolute_error(y_test, best_pred)
best_rmse = math.sqrt(mean_squared_error(y_test, best_pred))
best_r2 = r2_score(y_test, best_pred)

print("\nKNN Tốt Nhất")
print(f"K    : {best_k}")
print(f"MAE  : {best_mae:.3f}")
print(f"RMSE : {best_rmse:.3f}")
print(f"R²   : {best_r2:.3f}")

# =====================================================
# BƯỚC 9 - So sánh mô hình
# =====================================================

results = pd.DataFrame({
    "Model":[
        "Linear Regression",
        "KNN",
        "KNN + StandardScaler",
        f"KNN Best(K={best_k})"
    ],
    "MAE":[
        baseline_mae,
        knn_mae,
        scaled_mae,
        best_mae
    ],
    "RMSE":[
        baseline_rmse,
        knn_rmse,
        scaled_rmse,
        best_rmse
    ],
    "R2":[
        baseline_r2,
        knn_r2,
        scaled_r2,
        best_r2
    ]
})

print("\n============================")
print("BẢNG SO SÁNH")
print("============================")
print(results.round(3))

plt.figure(figsize=(8,5))
plt.bar(results["Model"], results["R2"])
plt.ylim(0,1)
plt.ylabel("R²")
plt.title("Model Comparison")

plt.savefig(
    "reports/figures/model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# =====================================================
# BƯỚC 10 - Phân tích lỗi theo nhóm tuổi
# =====================================================

error_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": best_pred
})

error_df["AgeGroup"] = pd.cut(
    error_df["Actual"],
    bins=[0,8,12,30],
    labels=["Young","Middle","Old"]
)

print("\nMAE theo nhóm tuổi")

group_mae = error_df.groupby("AgeGroup").apply(
    lambda x: mean_absolute_error(x["Actual"], x["Predicted"])
)

print(group_mae)

# =====================================================
# BƯỚC 11 - Actual vs Predicted
# =====================================================

plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    best_pred,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)

plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.title("Actual vs Predicted")

plt.savefig(
    "reports/figures/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# =====================================================
# BƯỚC 12 - Lưu Pipeline
# =====================================================

joblib.dump(
    best_pipeline,
    "models/abalone_knn_pipeline.pkl"
)

print("\nPipeline đã lưu:")
print("models/abalone_knn_pipeline.pkl")