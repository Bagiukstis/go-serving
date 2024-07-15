# Sklearn Machine Learning Model Serving with Python and Go

## About
This project involves training a simple machine learning (ML) model in Python, converting it to ONNX format, and serving it over HTTPS using a Golang server.

## Project Structure

- `golang/`: Contains Go server for serving the trained model.
- `images/`: Test image
- `models/`: Stored .onnx models
- `python/`: Contains Python script for training the ML model.

### Python
Trained sklearn ML model is converted to ONNX format using [skl2onnx converter](https://onnx.ai/sklearn-onnx/) and stored in the `models/` directory. <br>
1. **Create and activate a virtual environment (recommended):**

```bash
conda create --name [name] python=3.10
conda activate [name]
```

2. **Navigate to `python/` directory and install the requirements**
```bash
pip install -r requirements.txt
```
3. **Train and store the model**
```bash
python3 python/train.py
```

### Go
ONNX model is loaded using [onnxruntime_go wrapper](https://github.com/yalue/onnxruntime_go/) and served via HTTPS.<br>
For reproducing make sure you have `go` installed and run the following commands:
1. **Navigate to the Go directory**
```bash
cd golang
```
2. **Build the server binary**:
```bash
go build -o go-serving main.go
```
3. **Run the server**:
```bash
./go-serving
```

Open a new terminal and paste the following request:
```bash
curl -X POST http://localhost:8080/inference -d '{"input_data": [5.8, 2.8, 5.1, 2.4]}' -H "Content-Type: application/json"
```

Example input/output:
![alt text](images/test.png)