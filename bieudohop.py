import numpy as np
import matplotlib.pyplot as plt

# 1. Dữ liệu
data = [199, 201, 236, 269, 271, 278, 283, 291, 301, 303, 371]

# 2. Tính toán chính xác theo chuẩn SGK
# (Dùng percentile với method='lower' / 'higher' hoặc thủ công để khớp kết quả học viện)
Q1 = np.median(data[:5])
Q2 = np.median(data)
Q3 = np.median(data[6:])
IQR = Q3 - Q1

# Tính ranh giới xác định ngoại lai
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Lọc ngoại lai
outliers = [x for x in data if x < lower_bound or x > upper_bound]
print(f"Q1 = {Q1}")
print(f"Q2 = {Q2}")
print(f"Q3 = {Q3}")
print(f"IQR = {IQR}")
print(f"Giá trị ngoại lai: {outliers if outliers else 'Không có'}")

# 3. Vẽ biểu đồ hộp
plt.figure(figsize=(10, 4))
box = plt.boxplot(data, vert=False, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2))

# Thêm ghi chú
plt.title('Biểu đồ Hộp (Boxplot) Dữ Liệu', fontsize=14, pad=15)
plt.xlabel('Giá trị', fontsize=12)
plt.yticks([1], ['Tập dữ liệu'])

# Đánh dấu văn bản trên biểu đồ
plt.text(Q1, 1.1, f'Q1: {int(Q1)}', ha='center', color='blue')
plt.text(Q2, 0.9, f'Q2: {int(Q2)}', ha='center', color='red')
plt.text(Q3, 1.1, f'Q3: {int(Q3)}', ha='center', color='blue')

# Đường nét đứt để chỉ rõ cận cho ranh giới (Whisker bounds)
plt.axvline(lower_bound, color='gray', linestyle='--', alpha=0.7, label=f'Cận dưới ({lower_bound})')
plt.axvline(upper_bound, color='gray', linestyle='--', alpha=0.7, label=f'Cận trên ({upper_bound})')
plt.legend()

plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

