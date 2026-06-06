import numpy as np

# Khởi tạo ma trận A và B (Ví dụ kích thước 2x2)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 1. Phép cộng ma trận
C_add = A + B

# 2. Phép nhân ma trận
C_mul = np.dot(A, B)  # Hoặc có thể dùng A @ B

print("Ma trận A:\n", A)
print("Ma trận B:\n", B)
print("Kết quả phép cộng C = A + B:\n", C_add)
print("Kết quả phép nhân C = A x B:\n", C_mul)