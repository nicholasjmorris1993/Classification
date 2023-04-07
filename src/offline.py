import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight


def batching(df, outputs, test_frac):
    model = Batching()
    model.encode_labels(df, outputs)
    model.train(test_frac)
    model.predict()

    return model


class Batching:
    def encode_labels(self, df, outputs):
        self.data = df.copy().sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data
        self.outputs = outputs
        self.encoder = dict()

        for out in self.outputs:
            labels = self.data[out].tolist()
            self.encoder[out] = LabelEncoder()
            self.data[out] = self.encoder[out].fit_transform(labels)

    def train(self, test_frac):
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        self.model = dict()

        for out in self.outputs:
            X = train.copy().drop(columns=self.outputs)
            Y = train.copy()[[out]]

            classes_weights = class_weight.compute_sample_weight(
                class_weight="balanced",
                y=Y[out]
            )

            model = XGBClassifier(
                booster="gbtree",
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=7,
                min_child_weight=1,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=42,
            )
            model.fit(X, Y, sample_weight=classes_weights)

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
