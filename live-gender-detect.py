# Import Libraries
import cv2
import numpy as np

# Arsitektur model gender
GENDER_MODEL = "weights/deploy_gender.prototxt"

# Arsitektur model wajah
FACE_PROTO = "weights/deploy.prototxt.txt"

# Bobot model gender yang telah dilatih sebelumnya
GENDER_PROTO = "weights/gender_net.caffemodel"

# Bobot model wajah yang sudah dilatih sebelumnya
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# sebuah tuple yang berisi nilai rata-rata untuk setiap channel dalam gambar
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Mewakili 2 nama gender
GENDER_LIST = ["Male", "Female"]

# memuat model Caffe wajah
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# Memuat model prediksi gender
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

# Menginisialisasi ukuran frame
frame_width = 1280
frame_height = 720


def get_faces(frame, confidence_threshold=0.5):
    # mengubah frame menjadi blob agar siap sebagai input Neural Network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # mengatur gambar sebagai input untuk Neural Network
    face_net.setInput(blob)
    # melakukan inferensi dan mendapatkan prediksi
    output = np.squeeze(face_net.forward())
    # inisialisasi list hasil
    faces = []
    # Looping melalui wajah yang terdeteksi
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            )
            # mengubah ke integer
            start_x, start_y, end_x, end_y = box.astype(np.int64)
            # memperluas kotak sedikit
            start_x, start_y, end_x, end_y = (
                start_x - 10,
                start_y - 10,
                end_x + 10,
                end_y + 10,
            )
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # tambahkan ke dalam list
            faces.append((start_x, start_y, end_x, end_y))
    return faces
