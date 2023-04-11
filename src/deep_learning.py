import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def nnet(df, outputs, test_frac, deep):
    model = NNet()
    model.encode_labels(df, outputs)
    model.scale(test_frac)
    model.train(deep)
    model.predict()

    return model


class NNet:
    def encode_labels(self, df, outputs):
        self.data = df.copy().sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data
        self.outputs = outputs
        self.encoder = dict()

        for out in self.outputs:
            labels = self.data[out].tolist()
            self.encoder[out] = LabelEncoder()
            self.data[out] = self.encoder[out].fit_transform(labels)

    def scale(self, test_frac):
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))
        X = train.copy().drop(columns=self.outputs)

        # standardize the inputs to take on values between 0 and 1
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = self.data.copy().drop(columns=self.outputs)
        columns = X.columns
        X = scaler.transform(X)
        X = pd.DataFrame(X, columns=columns)

        Y = self.data.copy()[self.outputs]
        self.data = pd.concat([Y, X], axis="columns")

    def train(self, deep):
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        self.model = dict()

        if deep:
            layer = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
        else:
            layer = [128, 128]

        for out in self.outputs:
            X = train.copy().drop(columns=self.outputs)
            Y = train.copy()[[out]]

            model = MLPClassifier(
                max_iter=1000,
                hidden_layer_sizes=layer,
                learning_rate_init=0.001,
                learning_rate="adaptive",
                batch_size=16,
                random_state=42,
            )
            model.fit(X, Y.to_numpy().ravel().astype("int"))

            self.model[out] = model

    def predict(self):
        test = self.data.copy().tail(int(len(self.data)*self.test_frac))
        
        self.metric = dict()
        self.predictions = dict()

        for out in self.outputs:
            X = test.copy().drop(columns=self.outputs)
            Y = test.copy()[[out]]

            model = self.model[out]
            y_pred = model.predict(X)
            y_true = Y.to_numpy().ravel().astype("int")

            metric = accuracy_score(
                y_true=y_true, 
                y_pred=y_pred,
            )
            metric = f"Accuracy: {100 * round(metric, 4)}%"

            predictions = pd.DataFrame({
                "Actual": self.encoder[out].inverse_transform(y_true),
                "Predicted": self.encoder[out].inverse_transform(y_pred),
            })

            self.metric[out] = metric
            self.predictions[out] = predictions
