import math


class KNeighborsClassifier:

    def __init__(self, k):
        self.k = k
        self.X_train = []
        self.y_train = []

    def euclidean_distance(self, row1, row2):
        distance = 0
        for i in range(len(row1)):
            distance += (row1[i] - row2[i]) ** 2
        return math.sqrt(distance)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []

        for data in X_test:
            distances = []

            for i in range(len(self.X_train)):
                distance = self.euclidean_distance(data, self.X_train[i])
                distances.append((distance, self.y_train[i]))

            distances.sort(key=lambda x: x[0])

            neighbors = []
            for i in range(self.k):
                neighbors.append(distances[i][1])

            prediction = max(set(neighbors), key=neighbors.count)
            predictions.append(prediction)

        return predictions

    def score(self, y_pred, y_test):
        return (y_pred == y_test).mean()
