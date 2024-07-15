package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// EchoRequest represents the ingoing JSON request
type EchoRequest struct{
	InputData[] float32 `json:"input_data"`
}

// EchoResponse represents the outgoing JSON response
type EchoResponse struct {
    InputData[] float32 `json:"input_data"`
    Prediction[] int64 `json:"prediction"`
}

// Global variables for initializing the Sension, Input and Output tensors.
var (
    session       *ort.AdvancedSession
    inputTensor   *ort.Tensor[float32]
    outputTensor  *ort.Tensor[int64]
	mutex sync.Mutex
)

func getDefaultSharedLibPath() string {
	// For now, we only include libraries for x86_64 windows, ARM64 darwin, and
	// x86_64 or ARM64 Linux. In the future, libraries may be added or removed.
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.so"
		}
		return "./third_party/onnxruntime.so"
	}
	fmt.Printf("Unable to determine a path to the onnxruntime shared library"+
		" for OS \"%s\" and architecture \"%s\".\n", runtime.GOOS,
		runtime.GOARCH)
	return ""
}

func predict(inputData []float32) []int64{
	mutex.Lock()
	defer mutex.Unlock()

	// Getting current tensorData
	tensorData := inputTensor.GetData()

	// log.Println("Current tensorData ", tensorData)

	// Changing current tensor input to new one
	for i := range tensorData{
		tensorData[i] = inputData[i]
	}

	// log.Println("New tensorData ", tensorData)

	// Session inference
	if err := session.Run(); err != nil {
        log.Fatal(err)
    }

	// Prediction output
	return outputTensor.GetData()
}

func initModel(){
	// Set the shared library path for ORT
    ort.SetSharedLibraryPath(getDefaultSharedLibPath())
    
    // Initialize the ORT environment
    if err := ort.InitializeEnvironment(); err != nil {
        log.Fatalf("Failed to initialize ORT environment: %v", err)
    }

    // Prepare input tensor. Expected data shape is a vector: 1x4
    inputData := []float32{5.8, 2.8, 5.1, 2.4}
    inputShape := ort.NewShape(1, 4)
    var err error
    inputTensor, err = ort.NewTensor(inputShape, inputData)
    if err != nil {
        log.Fatalf("Failed to create input tensor: %v", err)
    }

    // Prepare output tensor. Shape -> 1
    outputShape := ort.NewShape(1)
    outputTensor, err = ort.NewEmptyTensor[int64](outputShape)
    if err != nil {
        log.Fatalf("Failed to create output tensor: %v", err)
    }

    // Load the model and create the session
    session, err = ort.NewAdvancedSession("../models/rf_iris.onnx",
        []string{"X"}, []string{"output_label"},
        []ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, nil)
    if err != nil {
        log.Fatalf("Failed to create session: %v", err)
    }
}

func cleanup() {
    if inputTensor != nil {
        inputTensor.Destroy()
    }
    if outputTensor != nil {
        outputTensor.Destroy()
    }
    if session != nil {
        session.Destroy()
    }
    ort.DestroyEnvironment() // Clean up the ORT environment
}

func echoHandler(w http.ResponseWriter, r *http.Request){
	var req EchoRequest

	// Decode the request body into the EchoRequest struct
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil{
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	inputData:= req.InputData

	// upperMessage := strings.ToUpper(req.Message)

	res := EchoResponse{
		InputData: inputData,
		//Prediction: outputTensor.GetData()
	}
	
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(res); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func inferenceHandler(w http.ResponseWriter, r *http.Request){
	var req EchoRequest

	// Decode the request body into the EchoRequest struct
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil{
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate input data
	if len(req.InputData) != 4 {
        http.Error(w, "Input data must be an array of four float32 values", http.StatusBadRequest)
        return
    }

	// Input data
	inputData:= req.InputData

	// Prediction
	pred:= predict(inputData)
	
	// Response
	res := EchoResponse{
		InputData: inputData,
		Prediction: pred,
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(res); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}

}

func StartServer() {
	log.Println("Initializing onnxruntime environment")

	initModel()
    
	log.Println("Starting the server at http://localhost:8080/")
	http.HandleFunc("/echo", echoHandler)
	http.HandleFunc("/inference", inferenceHandler)
    
    log.Println("Available endpoints: /echo, /inference")
    
    if err := http.ListenAndServe(":8080", nil); err != nil {
        log.Fatalf("Server failed to start: %v", err)
    }
	defer cleanup()
}