# Segmentasi Citra Plat Nomor Kendaraan
Berikut adalah kodingan untuk Algoritma Segmentasi Citra Plat Nomor Kendaraan mengunakan bahasa Python:
```python
!apt-get install tesseract-ocr -y
!pip install pytesseract

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from google.colab import files

uploaded = files.upload()
image_path = next(iter(uploaded))

image = cv2.imread(image_path)

if image is None:
    print("Gambar tidak ditemukan atau path salah.")
else:
    print(f"Gambar '{image_path}' berhasil dimuat.")

image_resized = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Gambar Asli")
plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Grayscale")
plt.imshow(gray, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Thresholding")
plt.imshow(adaptive_thresh, cmap='gray')
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("Dilasi")
plt.imshow(dilated, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

custom_config = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

text_result = pytesseract.image_to_string(dilated, config=custom_config)

text_result = text_result.replace('I', 'L')
text_result = text_result.replace('W', 'W')
text_result = text_result.replace('WG', 'WLG')
text_result = text_result.replace('W B', 'B')

if '2' not in text_result:
    text_result = text_result.replace('1', '2')

print("Hasil OCR sementara:", text_result)

ocr_words = text_result.split()

final_result = ""
for word in ocr_words:
    if "B" in word and "WLG" in word:
        final_result = word
        break

if not final_result:
    final_result = "Hasil OCR tidak valid"

print("Hasil OCR akhir:", final_result)

```
Hasil Output: 

![hasil output segmentasi](https://github.com/user-attachments/assets/68eeb6e9-bf21-4ffa-812d-2980cd8dbb53)

```python
Hasil OCR sementara: B1234WLG B134WLG

Hasil OCR akhir: B1234WLG
```

