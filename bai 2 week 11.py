import numpy as np

data = [199, 201, 236, 269, 271, 278, 283, 291, 301, 303, 371]

# Tính giá trị trung bình
mean = np.mean(data)

# Tính độ lệch chuẩn (ddof=0 tương đương chia n như Toán THPT)
std = np.std(data, ddof=0) 

print(f"Giá trị trung bình (x_bar) = {mean}")
print(f"Độ lệch chuẩn (s) = {std:.2f}")
print("-" * 30)

# Xét khoảng 2s
lower_2s = mean - 2 * std
upper_2s = mean + 2 * std
outliers_2s = [x for x in data if x < lower_2s or x > upper_2s]
print(f"Khoảng 2s: ({lower_2s:.2f}, {upper_2s:.2f})")
print(f"=> Giá trị bất thường (2s): {outliers_2s if outliers_2s else 'Không có'}")
print("-" * 30)

# Xét khoảng 3s
lower_3s = mean - 3 * std
upper_3s = mean + 3 * std
outliers_3s = [x for x in data if x < lower_3s or x > upper_3s]
print(f"Khoảng 3s: ({lower_3s:.2f}, {upper_3s:.2f})")
print(f"=> Giá trị bất thường (3s): {outliers_3s if outliers_3s else 'Không có'}")
