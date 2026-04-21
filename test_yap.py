from ultralytics import YOLO

# Modeli yükle
model = YOLO('best.pt')

# Testi çalıştır ve ekranda göster
results = model.predict(source='test.jpg', show=True, save=True)

# Pencerenin hemen kapanmaması için bekle
input("İşlem tamamlandı. Kapatmak için Enter'a bas...")