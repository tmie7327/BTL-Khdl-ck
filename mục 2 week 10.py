import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import cho Phân cụm & Tiền xử lý
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

# Import bổ sung cho bài toán Phân loại (Supervised)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# 1. TIỀN XỬ LÝ DỮ LIỆU
# ==========================================
# Khởi tạo dataset (có cả thông tin ID và Hiệu suất thực tế)
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Doanh thu': [3, 6, 9, 10, 12, 15, 18, 20, 25, 8, 14, 22],
    'Lợi nhuận': [1.1, 2.0, 3.2, 3.4, 3.9, 5.0, 6.2, 6.6, 8.2, 2.5, 4.5, 7.5],
    'Chi phí Marketing': [0.5, 0.8, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 1.0, 1.9, 3.2],
    'Số lượng khách hàng': [320, 450, 700, 750, 850, 1100, 1350, 1500, 1800, 600, 1000, 1650],
    'Hiệu suất': ['Kém', 'Kém', 'Bình thường', 'Bình thường', 'Bình thường', 'Tốt', 'Tốt', 'Tốt', 'Xuất sắc', 'Kém', 'Bình thường', 'Xuất sắc']
}
df_full = pd.DataFrame(data)

# Chỉ sử dụng 4 cột Feature. Bỏ hoàn toàn ID và Đánh giá (Hiệu suất)
X = df_full[['Doanh thu', 'Lợi nhuận', 'Chi phí Marketing', 'Số lượng khách hàng']]

# Bắt buộc chuẩn hóa dữ liệu (Z-score Normalization) do dùng distance-based
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 2. CHẠY CÁC MÔ HÌNH PHÂN CỤM
# ==========================================
# 2.1. K-Means Clustering (Elbow Method)
inertia = []
K_range = range(1, 9)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o', linestyle='--')
plt.title('Phương pháp K-Means: Elbow Method')
plt.xlabel('Số cụm (K)')
plt.ylabel('Mức độ phân tán (Inertia)')
plt.grid(True)
plt.show()

# Điểm Elbow cho thấy K=4 là hợp lý (có độ gẫy). Ta sẽ phân thành 4 cụm.
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled) # Phân thành nhãn (0, 1, 2, 3)

# 2.2. Hierarchical Clustering (Dendrogram)
plt.figure(figsize=(10, 5))
linked = linkage(X_scaled, method='ward')
dendrogram(linked, labels=df_full['ID'].values)
plt.title('Hierarchical Clustering: Biểu đồ cây Dendrogram')
plt.xlabel('ID Công ty')
plt.ylabel('Khoảng cách Eculidean (phương pháp Ward)')
plt.axhline(y=3, color='r', linestyle='--') # Đường cắt chia thành 4 nhánh
plt.show()

hierarchical_labels = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward').fit_predict(X_scaled)

# 2.3. DBSCAN
# eps: khoảng cách tối đa giữa 2 quan sát để gom thành cụm 
# min_samples: tối thiểu 2 điểm làm 1 cụm
dbscan = DBSCAN(eps=1.2, min_samples=2)
dbscan_labels = dbscan.fit_predict(X_scaled)
print("\n--- KẾT QUẢ TỪ DBSCAN ---")
print("Labels DBSCAN:", dbscan_labels)
# -> DBSCAN thường sẽ gán rất nhiều nhãn -1 (nhiễu), với số lượng dữ liệu chỉ 12 quan sát
# và sự phân tán ở dạng tuyến tính thì thuật toán đo lường Density (mật độ) như DBSCAN
# hoạt động khá yếu kém trong việc phân tách dữ liệu so với K-Means hay Hierarchical.

# ==========================================
# 3. TRỰC QUAN HÓA (PCA 2D)
# ==========================================
# Rút gọn từ 4 features xuống 2 chiều (components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

# Plot phân cụm K-Means
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='viridis', s=150)
plt.title("Nhóm bởi AI: K-Means (K=4)")
plt.xlabel('Thành phần chính 1')
plt.ylabel('Thành phần chính 2')
plt.legend(title="Cụm (Cluster)")

# Plot kết quả thật ban đầu
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df_full['Hiệu suất'], palette='Set1', s=150)
plt.title("Đánh giá thực tế (Con người)")
plt.xlabel('Thành phần chính 1')
plt.ylabel('Thành phần chính 2')
plt.legend(title="Hiệu suất")

plt.tight_layout()
plt.show()

# ==========================================
# 4. ĐÁNH GIÁ VÀ SO SÁNH (Silhouette Score)
# ==========================================
print("\n--- ĐÁNH GIÁ SILHOUETTE SCORE (K=4) ---")
print(f"Silhouette K-Means: {silhouette_score(X_scaled, kmeans_labels):.3f}")
print(f"Silhouette Hierarchical: {silhouette_score(X_scaled, hierarchical_labels):.3f}")
# Điểm Silhouette Score mô hình nào tiệm cận 1 hơn là hợp lý và ít phân mảnh hơn.
# Với bộ dữ liệu này, cả hai thường cho kết quả gần giống nhau do cấu trúc rõ ràng.

# ==========================================
# 5. GẮN NHÃN VÀ ĐỐI CHIẾU
# ==========================================
df_full['Cụm K-Means'] = kmeans_labels
df_full['Cụm Hierarchical'] = hierarchical_labels

compare_df = df_full[['ID', 'Doanh thu', 'Hiệu suất', 'Cụm K-Means']].sort_values(by='Doanh thu')

print("\n--- ĐỐI CHIẾU NHÓM CỦA AI VÀ ĐÁNH GIÁ CHUYÊN GIA ---")
print(compare_df.to_string(index=False))

# ==========================================
# 6. HỌC CÓ GIÁM SÁT (PHÂN LOẠI - THÊM TỪ PHẦN 2)
# ==========================================
print("\n" + "="*60)
print("PHẦN 2: HỌC CÓ GIÁM SÁT (PHÂN LOẠI)")
print("="*60)

# 6.1 Mã hóa nhãn (Label Encoding)
# Đa lớp (Multi-class)
label_mapping = {'Kém': 0, 'Bình thường': 1, 'Tốt': 2, 'Xuất sắc': 3}
y_multi = df_full['Hiệu suất'].map(label_mapping)

# 2 lớp (Binary Classification)
# Kém, Bình thường -> 0 (Kém) | Tốt, Xuất sắc -> 1 (Tốt)
binary_mapping = {'Kém': 0, 'Bình thường': 0, 'Tốt': 1, 'Xuất sắc': 1}
y_binary = df_full['Hiệu suất'].map(binary_mapping)

# 6.2 Chia tập huấn luyện / kiểm thử
# Do dataset nhỏ (12 dòng) ta dành ra khoảng 3 dòng để test
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X_scaled, y_binary, test_size=0.25, random_state=42)
_, _, y_train_mul, y_test_mul = train_test_split(X_scaled, y_multi, test_size=0.25, random_state=42)

# 6.3 Danh sách mô hình
models = {
    'Logistic Reg': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50),
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(kernel='linear', probability=True, random_state=42)
}

# 6.4 Đánh giá bài toán Binary (2 lớp)
print("\n[+] ĐÁNH GIÁ BÀI TOÁN 2 LỚP (Kém vs Tốt)")
results_bin = []
for name, model in models.items():
    model.fit(X_train, y_train_bin)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test_bin, y_pred)
    prec = precision_score(y_test_bin, y_pred, zero_division=0)
    rec = recall_score(y_test_bin, y_pred, zero_division=0)
    f1 = f1_score(y_test_bin, y_pred, zero_division=0)
    results_bin.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})

df_bin_res = pd.DataFrame(results_bin)
print(df_bin_res.to_string(index=False))

# 6.5 Đánh giá bài toán Multi-class (4 lớp)
print("\n[+] ĐÁNH GIÁ BÀI TOÁN ĐA LỚP (Kém, Bình thường, Tốt, Xuất sắc)")
results_mul = []
for name, model in models.items():
    model.fit(X_train, y_train_mul)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test_mul, y_pred)
    prec = precision_score(y_test_mul, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test_mul, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_mul, y_pred, average='weighted', zero_division=0)
    results_mul.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})

df_mul_res = pd.DataFrame(results_mul)
print(df_mul_res.to_string(index=False))

# 6.6 Trực quan hóa metrics
df_bin_melt = df_bin_res.melt(id_vars='Model', var_name='Metric', value_name='Score')
plt.figure(figsize=(10, 5))
sns.barplot(data=df_bin_melt, x='Model', y='Score', hue='Metric')
plt.title('So sánh hiệu suất các mô hình Phân loại (Bài toán Nhị phân)')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ==========================================
# 7. NÂNG CAO (TỰ VIẾT THUẬT TOÁN TỪ ĐẦU)
# ==========================================
print("\n" + "="*60)
print("PHẦN 4: NÂNG CAO (CUSTOM ALGORITHMS)")
print("="*60)

# Hàm Simple K-Means
def simple_kmeans(X, k=4, max_iters=100):
    np.random.seed(42) # Để kết quả tái lập được
    # Lấy ngẫu nhiên k điểm dữ liệu làm tâm ban đầu
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    
    for i in range(max_iters):
        # Tính khoảng cách Euclidean
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        # Gán nhãn cho tâm gần nhất
        labels = np.argmin(distances, axis=1)
        # Cập nhật tâm
        new_centroids = np.array([X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centroids[j] for j in range(k)])
        if np.allclose(centroids, new_centroids):
            print(f"-> Simple K-Means hội tụ tại vòng thứ {i+1}")
            break
        centroids = new_centroids
    return labels

# Do X_scaled là một 2D Numpy Array, có thể truyền trực tiếp
my_labels = simple_kmeans(X_scaled, k=4)
print("Nhãn dự đoán từ custom K-Means :", my_labels)
print("Nhãn dự đoán từ sklearn K-Means:", kmeans_labels)