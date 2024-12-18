import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

file_path = "../data/original.xlsx"
df = pd.read_excel(file_path)

X = df[["x1", "x2", "x3", "x4"]].values
y = df["ans"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

svr = SVR(kernel="rbf")
svr.fit(X_scaled, y)


def predict_svr(x1, x2, x3, x4):
    input_data = np.array([[x1, x2, x3, x4]])
    input_scaled = scaler.transform(input_data)
    return svr.predict(input_scaled)[0]
