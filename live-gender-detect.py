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
