# Train a model.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)
clr = RandomForestClassifier()
clr.fit(X_train, y_train)

# Convert into ONNX format.
from skl2onnx import to_onnx

onx = to_onnx(clr, X[:1])
with open("golang/models/rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# Compute the prediction with onnxruntime.
import onnxruntime as rt

sess = rt.InferenceSession("golang/models/rf_iris.onnx", providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]

print('Pred: ', pred_onx)
print('Ground truth: ', y_test)
print('Input tensor shape: ', X_train[0].shape)
print('Output tensor shape: ', y_train[0].shape)
print('Input_name: ', input_name)
print('Output_name: ', label_name)