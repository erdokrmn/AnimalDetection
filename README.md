# 🦁🕷️🐓 Animal Detection with YOLO11

Bu proje, **YOLO11s (Small)** modeli kullanılarak geliştirilmiş, özelleştirilmiş bir nesne tespit (object detection) sistemidir. Proje; **Fil, Örümcek ve Horoz** gibi çeşitli hayvan türlerini görüntüler üzerinden yüksek doğrulukla tespit etmek amacıyla eğitilmiştir.

Veri seti **Roboflow** üzerinde manuel olarak etiketlenmiş ve **Ultralytics** kütüphanesi ile eğitilmiştir.

---

## 📸 Proje Sonuçları (Demo)

Eğitilen modelin gerçek dünya verileri üzerindeki test sonuçları aşağıdadır:

| 🐘 Fil Tespiti | 🕷️ Örümcek Tespiti | 🐓 Horoz Tespiti |
| :---: | :---: | :---: |
| ![Fil Tespiti](FilTahmin.png) | ![Örümcek Tespiti](ÖrümcekTahmin.png) | ![Horoz Tespiti](HorozTahmin.png) |
| *Model fili başarıyla çerçeve içine alıyor.* | *Küçük ve karmaşık yapılı örümcek tespiti.* | *Horoz tespiti ve sınıflandırması.* |

---

## 🚀 Kullanılan Teknolojiler

* **Model:** [YOLO11s (Small)](https://github.com/ultralytics/ultralytics) - Hız ve doğruluk dengesi için seçildi.
* **Dil:** Python 3.11+
* **Veri Seti Yönetimi:** Roboflow
* **Kütüphaneler:** Ultralytics, OpenCV, Pillow

## 📂 Kurulum (Installation)

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1.  **Projeyi Klonlayın:**
    ```bash
    git clone [https://github.com/erdokrmn/AnimalDetection.git](https://github.com/erdokrmn/AnimalDetection.git)
    cd AnimalDetection
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install ultralytics opencv-python pillow
    ```

## 💻 Kullanım (Usage)

Modeli test etmek için aşağıdaki Python kodunu kullanabilirsiniz:

```python
from ultralytics import YOLO

# Eğitilmiş ağırlıkları yükle
model = YOLO("runs/detect/train/weights/best.pt")

# Bir resim üzerinde tahmin yap
results = model.predict("test_images/ornek_hayvan.jpg", save=True)
Veya terminal üzerinden doğrudan tahmin yapabilirsiniz:

Bash

yolo predict model=runs/detect/train/weights/best.pt source='test_video.mp4' show=True
📊 Eğitim Süreci (Training)
Model, Roboflow'dan çekilen veri seti üzerinde aşağıdaki parametrelerle eğitilmiştir:

Epoch: 50 (İsteğe bağlı artırılabilir)

Image Size: 640

Batch Size: 16

Optimizer: Auto

Eğitimi tekrar başlatmak isterseniz:

Python

from ultralytics import YOLO

model = YOLO("yolo11s.pt") # Pre-trained model
model.train(data="data.yaml", epochs=50, imgsz=640)
🤝 İletişim
Geliştirici: Erdinç Karaman

Bu proje hakkında sorularınız veya önerileriniz varsa GitHub üzerinden ulaşabilirsiniz.

Bu proje açık kaynaklıdır ve eğitim amaçlı geliştirilmiştir.
