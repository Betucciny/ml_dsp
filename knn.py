import numpy as np

class KNeighbors():
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=self.y.dtype)
        for i, x in enumerate(X):
            distances = np.sqrt(np.sum((self.X - x) ** 2, axis=1))
            k_idx = np.argsort(distances)[:self.k]
            k_neighbor_labels = self.y[k_idx]
            y_pred[i] = np.bincount(k_neighbor_labels).argmax()
        return y_pred

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


