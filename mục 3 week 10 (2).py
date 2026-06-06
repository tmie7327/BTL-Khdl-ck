import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore") # Tắt cảnh báo để output gọn gàng

# 1. CHUẨN BỊ DỮ LIỆU
data = {
    'Cong_ty': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Doanh_thu': [3, 6, 9, 10, 12, 15, 18, 20, 25, 8, 14, 22],
    'Loi_nhuan': [1.1, 2.0, 3.2, 3.4, 3.9, 5.0, 6.2, 6.6, 8.2, 2.5, 4.5, 7.5],
    'Chi_phi_Marketing': [0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 1.0, 1.9, 3.2],
    'So_luong_khach_hang': [320, 450, 700, 750, 850, 1100, 1350, 1500, 1800, 600, 1000, 1650],
    'Label': ['Kém', 'Kém', 'Bình thường', 'Bình thường', 'Bình thường', 'Tốt', 'Tốt', 'Tốt', 'Xuất sắc', 'Kém', 'Bình thường', 'Xuất sắc']
}
df = pd.DataFrame(data)

# Tách Features (Đặc trưng X) và Target (Nhãn Y)
X = df[['Doanh_thu', 'Loi_nhuan', 'Chi_phi_Marketing', 'So_luong_khach_hang']]
y = df['Label']

# ==========================================
# MÔ HÌNH 1: PHÂN LOẠI (CLASSIFICATION)
# ==========================================
print("--- 1. MÔ HÌNH PHÂN LOẠI (Decision Tree) ---")
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y) # Học từ dữ liệu

# Kiểm tra độ chính xác (do dữ liệu siêu nhỏ, accuracy sẽ thường là 100% nếu test trên cùng tập train)
y_pred = clf.predict(X)
print(f"Độ chính xác (Accuracy): {accuracy_score(y, y_pred) * 100}%")

# Bỏ thử một công ty mới tinh vào để máy dự đoán
new_company = [[16, 5.5, 2.2, 1200]] # Doanh thu 16, Lợi nhuận 5.5, CP 2.2, Khách 1200
print(f"Dự đoán cho công ty mới {new_company[0]}: -> {clf.predict(new_company)[0]}\n")


# ==========================================
# MÔ HÌNH 2: PHÂN CỤM (CLUSTERING)
# ==========================================
print("--- 2. MÔ HÌNH PHÂN CỤM (K-Means) ---")
# Cực kì quan trọng: Cần Scale (chuẩn hóa) biến khách hàng đang ở mức hàng ngàn về cùng thang đo với các chỉ số tỷ VNĐ.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sử dụng thuật toán K-Means chia thành 4 nhóm (vi mô hình ko biết trước label nên ta chỉ định số cụm K=4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster_ID'] = kmeans.fit_predict(X_scaled) 

# In ra để so sánh sự tương đồng giữa "Nhóm AI tự phân" với "Label người dán"
print("Đối chiếu Nhóm (Cluster) máy phân tích so với Label gốc:")
print(df[['Cong_ty', 'Doanh_thu', 'Label', 'Cluster_ID']])
