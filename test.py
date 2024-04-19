from unittest import result
from ultralytics import YOLO

model = YOLO('ultralytics/cfg/models/v3/yolov3.yaml')

results = model.train(data='coco128.yaml', epochs=3)



