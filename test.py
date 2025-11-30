import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os

class YOLOTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Model Test ve Karşılaştırma Aracı")
        self.root.geometry("1200x800")

        # --- AYARLAR ---
        # EĞİTTİĞİN MODELİN YOLUNU BURAYA GİR:
        model_path = "runs/detect/train3/weights/best.pt" 
        
        if not os.path.exists(model_path):
            messagebox.showerror("Hata", f"Model dosyası bulunamadı:\n{model_path}\nLütfen kodun içindeki yolu düzeltin.")
            root.destroy()
            return

        print(f"Model yükleniyor: {model_path}...")
        self.model = YOLO(model_path)
        print("Model yüklendi.")

        # Değişkenler
        self.current_image_path = None
        self.tk_image = None
        self.original_pil_image = None
        self.display_image_width = 0
        self.display_image_height = 0
        
        # Çizim değişkenleri
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.manual_boxes = [] # Manuel çizilen kutuları tutar

        # --- ARAYÜZ ELEMANLARI ---
        # Üst Panel (Butonlar)
        self.control_frame = tk.Frame(root, bg="lightgray", height=50)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        btn_load = tk.Button(self.control_frame, text="Resim Yükle (Örümcek, Fil vb.)", command=self.load_image, bg="lightblue", font=("Arial", 12))
        btn_load.pack(side=tk.LEFT, padx=10, pady=10)

        btn_clear_boxes = tk.Button(self.control_frame, text="Manuel Çizimleri Temizle", command=self.clear_manual_boxes, font=("Arial", 10))
        btn_clear_boxes.pack(side=tk.LEFT, padx=10, pady=10)

        btn_detect = tk.Button(self.control_frame, text="MODELİ TEST ET (Tahmin Yap)", command=self.run_detection, bg="orange", font=("Arial", 12, "bold"))
        btn_detect.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.info_label = tk.Label(self.control_frame, text="1. Resim Yükle -> 2. Mouse ile kendi etiketini çiz (Mavi) -> 3. Modeli Test Et", bg="lightgray")
        self.info_label.pack(side=tk.RIGHT, padx=10)

        # Ana Canvas (Resim Alanı)
        self.canvas_frame = tk.Frame(root, bg="darkgray")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, bg="gray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Mouse olaylarını bağla (Manuel çizim için)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not file_path:
            return

        self.current_image_path = file_path
        self.original_pil_image = Image.open(file_path)
        
        # Resmi canvas'a sığacak şekilde yeniden boyutlandır
        self.display_image(self.original_pil_image)
        self.clear_manual_boxes()
        print(f"Resim yüklendi: {file_path}")

    def display_image(self, pil_image):
        # Canvas boyutlarını al
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 100 or canvas_height < 100: # İlk açılışta çok küçükse varsayılan
             canvas_width = 1100
             canvas_height = 700

        # Görüntü oranını koruyarak yeniden boyutlandırma hesaplaması
        img_width, img_height = pil_image.size
        ratio = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        self.display_image_width = new_width
        self.display_image_height = new_height

        resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)

        self.canvas.delete("all") # Canvası temizle
        # Resmi merkeze yerleştir
        self.canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=self.tk_image, tags="main_image")

    # --- MANUEL ÇİZİM FONKSİYONLARI ---
    def on_button_press(self, event):
        if self.tk_image is None: return
        self.start_x = event.x
        self.start_y = event.y
        # Geçici bir dikdörtgen oluştur (Mavi renk)
        self.current_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='blue', width=3, tags="manual_box")

    def on_move_press(self, event):
        if self.current_rect:
            self.canvas.coords(self.current_rect, self.start_x, self.start_y, event.x, event.y)

    def on_button_release(self, event):
        if self.current_rect:
            # Çizilen kutunun koordinatlarını sakla (Gerekirse ileride karşılaştırma metriği için kullanılabilir)
            box_coords = self.canvas.coords(self.current_rect)
            self.manual_boxes.append(box_coords)
            self.current_rect = None

    def clear_manual_boxes(self):
        self.canvas.delete("manual_box")
        self.manual_boxes = []

    # --- YOLO TAHMİN FONKSİYONU ---
    def run_detection(self):
        if self.current_image_path is None:
            messagebox.showwarning("Uyarı", "Lütfen önce bir resim yükleyin.")
            return
        
        print("Tahmin yapılıyor...")
        # 1. Modeli çalıştır
        results = self.model.predict(source=self.current_image_path, conf=0.25) # Conf eşiğini değiştirebilirsin

        # 2. Sonuç görselini al (Ultralytics çizimi yapar)
        result_bgr_image = results[0].plot() # OpenCV formatında döner (BGR)

        # 3. Renk formatını düzelt (BGR -> RGB)
        result_rgb_image = cv2.cvtColor(result_bgr_image, cv2.COLOR_BGR2RGB)

        # 4. PIL formatına çevir
        pil_result_image = Image.fromarray(result_rgb_image)

        # 5. Ekrana bas
        # Amaç modelin ne gördüğünü görmektir.
        self.display_image(pil_result_image)
        
        messagebox.showinfo("Tamamlandı", f"Tahmin tamamlandı.\nTespit edilen nesne sayısı: {len(results[0].boxes)}")


if __name__ == "__main__":
    # Windows multiprocessing hatası için önlem (Eğer predict içinde gerekirse diye)
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except ImportError:
        pass
        
    root = tk.Tk()
    app = YOLOTestApp(root)
    # Pencere açıldığında tam boyut güncellemesi için kısa bir gecikme
    root.after(100, lambda: app.display_image(app.original_pil_image) if app.original_pil_image else None)
    root.mainloop()