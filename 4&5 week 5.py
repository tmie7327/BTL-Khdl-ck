import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 5. Tái tạo dữ liệu từ ảnh
data = {
    'Doanh_thu_x1': [3e8, 6e8, 9e8, 10e8, 12e8, 15e8, 18e8, 20e8],
    'Chi_phi_MKT_x2': [5e7, 1e8, 1.5e8, 2e8, 2.5e8, 3e8, 3.5e8, 4e8],
    'Chi_luong_x3': [1e8, 1.2e8, 1.5e8, 1.8e8, 2e8, 2.2e8, 2.5e8, 2.8e8],
    'So_nhan_su_x4': [10, 12, 15, 18, 20, 22, 25, 28],
    'Loi_nhuan_y': [1.1e8, 2e8, 3.2e8, 3.4e8, 3.9e8, 5e8, 6.2e8, 6.6e8]
}
df = pd.DataFrame(data)

X = df.drop('Loi_nhuan_y', axis=1)
y = df['Loi_nhuan_y']

# Chuẩn hóa dữ liệu (Cực kỳ quan trọng)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train model với thư viện
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

print(f"R-squared: {model.score(X_test, y_test)}")