from ultralytics import YOLO

model = YOLO('mthong_detect/yolov9_vietsol/models/model_weights/collection/yolov9c.pt')

model.info(detailed=True)

