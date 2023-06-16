import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

from preprocessing import genres
from knn import KNeighbors
from convolutional import ConvolutionalNN


genres_dict = { genre: i for i, genre in enumerate(genres) }
tf.config.experimental.set_visible_devices([], 'GPU')


def conffusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    n_classes = len(genres)
    matrix = np.zeros((n_classes, n_classes))
    for i in range(len(y_true)):
        matrix[y_true[i]][y_pred[i]] += 1
    sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=genres, yticklabels=genres)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


def normalize(data):
    return (data - np.mean(data)) / np.std(data)


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    newX = X.to_numpy()
    newy = [genres_dict[genre] for genre in y]
    newy = np.array(newy)
    train_idx = np.random.choice(len(newX), round(len(newX) * (1-test_size)), replace=False)
    test_idx = np.array(list(set(range(len(newX))) - set(train_idx)))
    X_train = newX[train_idx]
    y_train = newy[train_idx]
    X_test = newX[test_idx]
    y_test = newy[test_idx]
    return X_train, X_test, y_train, y_test


def main():
    data_frame = pd.DataFrame()
    for genre in genres:
        temp = pd.read_csv(f'./Data/{genre}.csv')
        data_frame = pd.concat([data_frame, temp], axis=0)
    data_frame = data_frame.drop(columns=['file_name'])
    X = data_frame.drop(columns=['label'])
    y = data_frame['label']
    X = X.apply(normalize, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Modelo de KNN
    knn = KNeighbors(k=5)
    knn.fit(X_train, y_train)
    print("KNN Score:", knn.score(X_test, y_test))
    conffusion_matrix(y_test, knn.predict(X_test), title="KNN Confusion Matrix")

    # Modelo de redes neuronales
    cnn = ConvolutionalNN(X.shape[1], 10)
    history = cnn.fit(X_train, y_train, verbose=0, epochs=300, validation_data=(X_test, y_test))
    print("CNN Score:", cnn.score(X_test, y_test))
    conffusion_matrix(y_test, cnn.predict(X_test), title="CNN Confusion Matrix")
    pd.DataFrame(history.history).plot(figsize=(12, 6))
    plt.show()
    cnn.save('model.h5')


if __name__ == '__main__':
    main()
