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

# Represents 2 gender names
GENDER_LIST = ["Male", "Female"]

# memuat model Caffe wajah
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)

# Memuat model prediksi gender
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

# Menginisialisasi ukuran frame
frame_width = 1280
frame_height = 720


def get_faces(frame, accuracy_threshold=0.5):
    # mengubah frame menjadi blob agar siap sebagai input Neural Network
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # mengatur gambar sebagai input untuk Neural Network
    face_net.setInput(blob)
    # melakukan inferensi dan mendapatkan prediksi
    output = np.squeeze(face_net.forward())
    # inisialisasi variabel list faces
    faces = []
    # Looping melalui wajah yang terdeteksi
    for i in range(output.shape[0]):
        accuracy = output[i, 2]
        if accuracy > accuracy_threshold:
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


def get_optimal_font_scale(text, width):
    """Tentukan skala font yang optimal berdasarkan lebar bingkai hosting"""
    # Mengevaluasi skala font dari yang terbesar hingga terkecil
    for scale in reversed(range(0, 60, 1)):
        # Menghitung ukuran teks dengan menggunakan cv2.getTextSize()
        textSize = cv2.getTextSize(
            text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale / 10, thickness=1
        )
        new_width = textSize[0][0]
        # Memeriksa apakah lebar teks yang dihasilkan sesuai dengan lebar frame yang diinginkan
        if new_width <= width:
            # Jika sesuai, mengembalikan skala font yang optimal
            return scale / 10
    # Jika tidak ada skala yang sesuai, mengembalikan skala default yaitu 1
    return 1


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # inisialisasi dimensi gambar yang akan diubah ukurannya dan
    # dapatkan ukuran gambar
    dim = None
    (h, w) = image.shape[:2]
    # jika kedua lebar dan tinggi tidak ditentukan, maka kembalikan gambar asli
    if width is None and height is None:
        return image
    # periksa apakah lebar tidak ditentukan
    if width is None:
        # hitung rasio tinggi dan konstruksi dimensi
        r = height / float(h)
        dim = (int(w * r), height)
    # sebaliknya, jika tinggi tidak ditentukan
    else:
        # hitung rasio lebar dan konstruksi dimensi
        r = width / float(w)
        dim = (width, int(h * r))
    # ubah ukuran gambar
    return cv2.resize(image, dim, interpolation=inter)


def predict_gender():
    """Memprediksi jenis kelamin dari wajah yang ditampilkan dalam gambar"""

    # Membuat objek kamera baru
    capture = cv2.VideoCapture(0)

    while True:
        _, img = capture.read()
        # ubah ukuran gambar, uncomment jika Anda ingin mengubah ukuran gambar
        # img = cv2.resize(img, (frame_width, frame_height))
        # Salin gambar awal dan ubah ukurannya
        frame = img.copy()

        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)

        # Prediksi wajah
        faces = get_faces(frame)

        # Menginisialisasi jumlah wajah male dan female
        male_count = 0
        female_count = 0

        # Melakukan loop pada setiap wajah yang terdeteksi
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y:end_y, start_x:end_x]

            blob = cv2.dnn.blobFromImage(
                image=face_img,
                scalefactor=1.0,
                size=(227, 227),
                mean=MODEL_MEAN_VALUES,
                swapRB=False,
                crop=False,
            )

            # Prediksi Jenis Kelamin
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            i = gender_preds[0].argmax()
            gender = GENDER_LIST[i]
            gender_confidence_score = gender_preds[0][i]

            # Gambar persegi panjang
            label = "{}-{:.2f}%".format(gender, gender_confidence_score * 100)

            # Menambahkan jumlah wajah male dan female
            if gender == "Male":
                male_count += 1
                print(male_count, label)
            else:
                female_count += 1
                print(female_count, label)

            yPos = start_y - 15
            while yPos < 15:
                yPos += 15

            # Dapatkan skala font yang optimal untuk ukuran gambar ini
            optimal_font_scale = get_optimal_font_scale(label, ((end_x - start_x) + 25))
            box_color = (0, 128, 0) if gender == "Male" else (147, 20, 255)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)

            # Label gambar yang telah diproses
            cv2.putText(
                frame,
                label,
                (start_x, yPos),
                cv2.FONT_HERSHEY_SIMPLEX,
                optimal_font_scale,
                box_color,
                2,
            )

        # Tampilkan gambar yang telah diproses
        cv2.imshow("Predict Gender v1.0", frame)

        if cv2.waitKey(1) == ord("q"):
            break

        # uncomment jika kamu ingin menyimpan gambar
        # cv2.imwrite("output/live_output.jpg", frame)

    # Membersihkan jendela yang terbuka
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict_gender()
