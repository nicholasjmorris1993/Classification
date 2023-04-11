import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
sys.path.append("/home/nick/Classification/src")
from deep_learning import nnet


data = pd.read_csv("/home/nick/Classification/test/LungCap.csv")

model = nnet(
    df=data, 
    outputs=["LungCap"], 
    test_frac=0.5,
    deep=False,
)

print(model.metric[model.outputs[0]])

predictions = model.predictions[model.outputs[0]].copy()

labels = np.unique(predictions.to_numpy())
cmatrix = confusion_matrix(
    y_true=predictions["Actual"],   # rows
    y_pred=predictions["Predicted"],  # columns
    labels=labels,
)
cmatrix = pd.DataFrame(cmatrix, columns=labels, index=labels)
print(cmatrix)
