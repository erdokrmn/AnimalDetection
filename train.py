from ultralytics import YOLO

# Windows kullanıyorsan bu satır ZORUNLUDUR
if __name__ == '__main__':
    # Modeli yükle
    model = YOLO("yolo11s.pt")  # veya yolov8n.pt

    # Eğitimi başlat
    results = model.train(data="data.yaml", epochs=50, imgsz=640)