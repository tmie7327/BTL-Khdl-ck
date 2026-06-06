import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 1. Khởi tạo dữ liệu
data = {
    'Doanh_thu': [3, 6, 9, 10, 12, 15, 18, 20],
    'Loi_nhuan': [1.1, 2, 3.2, 3.4, 3.9, 5, 6.2, 6.6]
}
df = pd.DataFrame(data)

X = df[['Doanh_thu']].values
y = df['Loi_nhuan'].values

# Chia tập Train (6 điểm đầu) và Test (2 điểm tiếp theo)
X_train, y_train = X[:6], y[:6]
X_test, y_test = X[6:8], y[6:8]

# 2. So sánh các bậc đa thức từ 1 đến 4
degrees = [1, 2, 3, 4]
results = []

plt.figure(figsize=(12, 5))

# Biểu đồ bên trái: Trực quan hóa đường cong
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')

X_plot = np.linspace(0, 25, 100).reshape(-1, 1)

for degree in degrees:
    # Biến đổi dữ liệu sang bậc cao
    poly = PolynomialFeatures(degree=degree)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    
    # Dự báo trên tập Test để tính RMSE
    y_pred_test = model.predict(X_poly_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results.append({'Degree': degree, 'RMSE': rmse})
    
    # Vẽ đường cong dự báo
    plt.plot(X_plot, model.predict(poly.transform(X_plot)), label=f'Bậc {degree}')

plt.title('Đường cong dự báo theo bậc đa thức')
plt.xlabel('Doanh thu')
plt.ylabel('Lợi nhuận')
plt.legend()

# Biểu đồ bên phải: So sánh RMSE
plt.subplot(1, 2, 2)
res_df = pd.DataFrame(results)
plt.bar(res_df['Degree'].astype(str), res_df['RMSE'], color='orange')
plt.title('So sánh RMSE (Càng thấp càng tốt)')
plt.xlabel('Bậc đa thức')
plt.ylabel('RMSE')

print("BẢNG SO SÁNH SAI SỐ RMSE:")
print(res_df)
plt.tight_layout()
plt.show()