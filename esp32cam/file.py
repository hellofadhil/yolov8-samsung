import camera
import os

# Inisialisasi kamera
camera.init(0)  # 0 = Mode default
camera.framesize(camera.FRAME_VGA)  # Resolusi VGA
camera.quality(10)  # Kualitas gambar (1-63, semakin kecil semakin baik)
camera.flip(0)  # 0 = Tidak membalik gambar, 1 = Flip Vertikal, 2 = Flip Horizontal
camera.mirror(0)  # Mirror mode

# Ambil gambar
img = camera.capture()

# Simpan gambar ke file
file_name = "/flash/photo.jpg"
with open(file_name, "wb") as f:
    f.write(img)

print(f"Gambar berhasil disimpan di {file_name}")
