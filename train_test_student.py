import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# 1. Đọc dữ liệu
data_path = r"C:\Users\TRA MI\Downloads\student-mat.csv"
df = pd.read_csv(data_path, sep=";")

# 2. Tách feature và target
X = df.drop(columns=["G3"])
y = df["G3"]

# 3. Xác định cột phân loại (object) và cột số
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# 4. Tiền xử lý
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# 5. Mô hình
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
)

# 6. Pipeline = tiền xử lý + mô hình
clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model),
])

# 7. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Huấn luyện
clf.fit(X_train, y_train)

# 9. Đánh giá
y_pred = clf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE (sai số tuyệt đối trung bình): {mae:.2f}")
print(f"R² (độ phù hợp mô hình): {r2:.3f}")

# 10. Thử dự đoán cho 5 học sinh bất kỳ
sample = X_test.iloc[:5]
sample_true = y_test.iloc[:5]
sample_pred = clf.predict(sample)

print("\nDự đoán cho 5 học sinh đầu tiên trong tập test:")
for i in range(len(sample)):
    print(f"Học sinh {i+1}: G3 thực tế = {sample_true.iloc[i]}, dự đoán = {sample_pred[i]:.2f}")