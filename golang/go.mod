module go-serving/main

go 1.22.5

replace go-serving/server => ./server

require go-serving/server v0.0.0-00010101000000-000000000000

require github.com/yalue/onnxruntime_go v1.10.0 // indirect

replace github.com/yalue/onnxruntime_go => github.com/yalue/onnxruntime_go v1.9.0
