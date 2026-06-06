import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 1. Chuẩn bị dữ liệu từ bảng
data = {
    'Doanh_thu': [3, 6, 9, 10, 12, 15, 18, 20],
    'Loi_nhuan': [1.1, 2, 3.2, 3.4, 3.9, 5, 6.2, 6.6]
}
df = pd.DataFrame(data)

# X: Doanh thu, y: Lợi nhuận
X = df[['Doanh_thu']].values
y = df['Loi_nhuan'].values

# Chia tập theo yêu cầu: 6 dòng đầu để Train, 2 dòng tiếp theo để Test
X_train, y_train = X[:6], y[:6]
X_test, y_test = X[6:8], y[6:8]

# 2. So sánh RMSE qua các bậc đa thức
results = []
degrees = [1, 2, 3, 4]

print(f"{'Bậc':<10} | {'RMSE (Tập Test)':<20}")
print("-" * 35)

for d in degrees:
    # Tạo đặc trưng đa thức
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Dự báo trên tập Test
    y_pred = model.predict(X_test_poly)
    
    # Tính toán RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append(rmse)
    print(f"Bậc {d:<6} | {rmse:<20.4f}")

# 3. Kết luận
best_degree = degrees[np.argmin(results)]
print("-" * 35)
print(f"==> Bậc đa thức tốt nhất dựa trên RMSE là: Bậc {best_degree}")
