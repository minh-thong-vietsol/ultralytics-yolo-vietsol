from ultralytics import YOLO

model = YOLO('mthong_detect/yolov9_vietsol/models/model_weights/collection/yolov9c.pt')

source = 'mthong_detect/yolov9_vietsol/data/vinfast_data/Ngay3-720p.mp4'
max_det = 1000
save = True
save_txt = True

model.predict(source = source, max_det = max_det, save = save, save_txt = save_txt)

