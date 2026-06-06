import numpy as np

def gradient_descent(X, y, lr=0.01, epochs=1000):
    n = float(len(y))
    w = 0
    b = 0
    for i in range(epochs):
        y_pred = w * X + b
        # Đạo hàm MSE
        dw = (-2/n) * sum(X * (y - y_pred))
        db = (-2/n) * sum(y - y_pred)
        # Cập nhật
        w = w - lr * dw
        b = b - lr * db
    return w, b