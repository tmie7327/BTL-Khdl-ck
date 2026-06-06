from sklearn.decomposition import PCA

# Giảm từ 4 chiều (x1, x2, x3, x4) xuống còn 2 chiều để trực quan hóa
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Tỉ lệ phương sai giữ lại:", pca.explained_variance_ratio_)