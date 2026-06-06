def linear_regression_l2(X, y, lr=0.001, lambd=0.1, epochs=1000):
    w = np.zeros(X.shape[1])
    b = 0
    m = len(y)
    for _ in range(epochs):
        y_hat = np.dot(X, w) + b
        # Gradient có thêm thành phần Regularization 2*lambd*w
        dw = (1/m) * (np.dot(X.T, (y_hat - y)) + lambd * w)
        db = (1/m) * np.sum(y_hat - y)
        w -= lr * dw
        b -= lr * db
    return w, b