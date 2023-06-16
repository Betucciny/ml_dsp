from tensorflow import keras


class ConvolutionalNN():
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = keras.models.Sequential([
            keras.layers.Dense(512, activation="relu", input_shape=[self.input_shape,]),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softplus"),

        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X, y, epochs=300, batch_size=128, verbose=1, validation_data=None):
        return self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=validation_data)

    def predict(self, X):
        y_prob = self.model.predict(X)
        y_pred = y_prob.argmax(axis=-1)
        return y_pred

    def predict_proba(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=2)[1]

    def summary(self):
        return self.model.summary()

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = keras.models.load_model(path)
