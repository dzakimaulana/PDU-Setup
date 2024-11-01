import cv2
import numpy as np
import paho.mqtt.client as mqtt

broker_address = "localhost"
port = 8883
topic = "test/camera"

client = mqtt.Client()
client.connect(broker_address, port)

# Konfigurasi
lebar_minimal = 80  # Lebar minimum dari persegi panjang
tinggi_minimal = 80  # Tinggi minimum dari persegi panjang

offset = 6  # Kesalahan yang diizinkan antar pixel
posisi_garis = 550  # Posisi garis penghitungan
delay = 60  # FPS dari video

deteksi = []
jumlah_batu = 0

# Fungsi untuk mendapatkan titik pusat dari kontur
def ambil_pusat(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Menggunakan kamera laptop sebagai input
cap = cv2.VideoCapture(0)  # Indeks 0 biasanya untuk kamera laptop default
subtraksi_latar_belakang = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame1 = cap.read()
    if not ret:
        print("Gagal mengambil gambar dari kamera.")
        break

    abu_abu = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(abu_abu, (3, 3), 5)
    img_sub = subtraksi_latar_belakang.apply(blur)
    dilasi = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilasi = cv2.morphologyEx(dilasi, cv2.MORPH_CLOSE, kernel)
    dilasi = cv2.morphologyEx(dilasi, cv2.MORPH_CLOSE, kernel)
    kontur, _ = cv2.findContours(dilasi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Kotak penghitungan untuk deteksi batu
    kotak_penghitungan = (55, 300, 500, 450)
    cv2.rectangle(frame1, (55, 300), (500, 450), (255, 127, 0), 3)

    for c in kontur:
        x, y, w, h = cv2.boundingRect(c)
        validasi_kontur = np.all([w >= lebar_minimal, h >= tinggi_minimal])
        if not validasi_kontur:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        pusat = ambil_pusat(x, y, w, h)
        deteksi.append(pusat)
        cv2.circle(frame1, pusat, 4, (0, 0, 255), -1)

    for (cx, cy) in deteksi:
        if kotak_penghitungan[0] < cx < kotak_penghitungan[2] and kotak_penghitungan[1] < cy < kotak_penghitungan[3]:
            jumlah_batu += 1
            print("Ada batu")
            cv2.rectangle(frame1, (55, 300), (500, 450), (0, 127, 255), 3)
            deteksi.remove((cx, cy))
            client.publish(topic, jumlah_batu)
            print("Batu terdeteksi: " + str(jumlah_batu))

    cv2.putText(frame1, "JUMLAH BATU : " + str(jumlah_batu), (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video Asli", frame1)
    cv2.imshow("Deteksi", dilasi)

    if cv2.waitKey(1) == 27:  # Tekan 'Esc' untuk keluar
        break

cv2.destroyAllWindows()
cap.release()
