import pandas as pd
from river import stream
from river import forest
from river import metrics


def streaming(df, outputs, test_frac):
    model = Streaming()
    model.train(df, outputs, test_frac)
    model.predict()

    return model


class Streaming:
    def train(self, df, outputs, test_frac):
        self.data = df.copy().sample(frac=1, random_state=0).reset_index(drop=True)  # shuffle the data
        self.outputs = outputs
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        self.model = dict()

        for out in self.outputs:
            X = train.copy().drop(columns=self.outputs)
            Y = train.copy()[[out]]

            model = forest.ARFClassifier(n_models=10, seed=42)

            for x, y in stream.iter_pandas(X, Y):
                model = model.learn_one(x, y[out])

            self.model[out] = model

    def predict(self):
        test = self.data.copy().tail(int(len(self.data)*self.test_frac))
        
        self.metric = dict()
        self.predictions = dict()

        for out in self.outputs:
            X = test.copy().drop(columns=self.outputs)
            Y = test.copy()[[out]]

            model = self.model[out]
            metric = metrics.Accuracy()
            predictions = pd.DataFrame()

            for x, y in stream.iter_pandas(X, Y):
                y_pred = model.predict_one(x)
                metric = metric.update(y[out], y_pred)
                model = model.learn_one(x, y[out])

                pred = pd.DataFrame({
                    "Actual": [y[out]],
                    "Predicted": [y_pred],
                })
                predictions = pd.concat([
                    predictions, 
                    pred,
                ], axis="index").reset_index(drop=True)

            self.model[out] = model
            self.metric[out] = metric
            self.predictions[out] = predictions
