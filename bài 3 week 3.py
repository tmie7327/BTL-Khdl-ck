"""
regression_and_matrix.py

Contains:
- matrix addition and multiplication (pure Python)
- simple linear regression y = a*x + b (normal equations)
- plotting and RMSE calculation

Run with: python regression_and_matrix.py
"""

import math

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except Exception:
    plt = None
    PLOT_AVAILABLE = False


def mat_add(A, B):
    """Pure-Python matrix addition. A and B are lists of lists."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def mat_mul(A, B):
    """Pure-Python matrix multiplication. A: m x p, B: p x n."""
    m = len(A)
    p = len(B)
    n = len(B[0])
    C = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C


def linear_regression_normal_eq(x, y):
    """Compute linear regression coefficients a, b for y = a*x + b via normal equations."""
    n = len(x)
    if n == 0:
        raise ValueError("Empty data")
    Sx = sum(x)
    Sy = sum(y)
    Sxx = sum(xi * xi for xi in x)
    Sxy = sum(xi * yi for xi, yi in zip(x, y))
    denom = n * Sxx - Sx * Sx
    if abs(denom) < 1e-12:
        raise ValueError("Denominator too small for stable fit")
    a = (n * Sxy - Sx * Sy) / denom
    b = (Sy - a * Sx) / n
    return a, b


def rmse(y_true, y_pred):
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    if n == 0:
        return float('nan')
    s = 0.0
    for yt, yp in zip(y_true, y_pred):
        d = yt - yp
        s += d * d
    return math.sqrt(s / n)


def main():
    # Dữ liệu (Doanh thu -> Lợi nhuận)
    x = [3, 6, 9, 10, 12, 15, 18, 20]
    y = [1.1, 2, 3.2, 3.4, 3.9, 5, 6.2, 6.6]

    a, b = linear_regression_normal_eq(x, y)
    y_pred = [a * xi + b for xi in x]
    train_rmse = rmse(y, y_pred)

    # Dự đoán tháng 9 với doanh thu = 25
    x9 = 25.0
    y9 = a * x9 + b

    print(f"Linear model: y = {a:.6f} * x + {b:.6f}")
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Dự đoán Lợi nhuận tháng 9 (x=25): {y9:.4f}")

    if PLOT_AVAILABLE:
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, label='Dữ liệu (Doanh thu, Lợi nhuận)')
        xs = [min(x) + i * (x9 - min(x)) / 199 for i in range(200)]
        plt.plot(xs, [a * xi + b for xi in xs], 'r-', label=f'Hồi quy: y={a:.4f}x+{b:.4f}')
        plt.scatter([x9], [y9], c='green', label=f'Dự đoán tháng 9 = {y9:.2f}')
        plt.xlabel('Doanh thu')
        plt.ylabel('Lợi nhuận')
        plt.title('Hồi quy tuyến tính (Bình phương tối thiểu)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print('Plot unavailable because matplotlib/numpy is not installed or cannot be imported.')


if __name__ == '__main__':
    main()
