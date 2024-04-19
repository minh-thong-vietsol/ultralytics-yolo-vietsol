from os import path
from unittest import result
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v3/yolov3.yaml')

#Valid formats are 
#('torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle', 'ncnn')

model.export(format='onnx')



