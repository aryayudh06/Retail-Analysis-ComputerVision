import os
import cv2
import numpy as np
from tqdm import tqdm

# Path ke dataset UTKFace
UTKFACE_PATH = "face_test"  # Ganti dengan path sebenarnya

# Load model
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Konstanta
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)',
           '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
age_to_category = {
    '(0-2)': 'kid',
    '(4-6)': 'kid',
    '(8-12)': 'kid',
    '(15-20)': 'teen',
    '(25-32)': 'adult',
    '(38-43)': 'adult',
    '(48-53)': 'adult',
    '(60-100)': 'elder'
}

def get_label_from_filename(filename):
    try:
        age, gender, *_ = filename.split("_")
        age = int(age)
        gender = int(gender)
        # Mapping umur ke kategori
        if age <= 12:
            age_cat = "kid"
        elif age <= 20:
            age_cat = "teen"
        elif age <= 53:
            age_cat = "adult"
        else:
            age_cat = "elder"
        return genderList[gender], age_cat
    except:
        return None, None

def predict_gender_age(img):
    blob = cv2.dnn.blobFromImage(img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    age_cat = age_to_category.get(age, "unknown")

    return gender, age_cat

# Inisialisasi akurasi
total, correct_gender, correct_age = 0, 0, 0

# Loop semua file gambar
for file in tqdm(os.listdir(UTKFACE_PATH)):
    if not file.endswith(".jpg"):
        continue

    true_gender, true_age_cat = get_label_from_filename(file)
    if true_gender is None:
        continue

    img_path = os.path.join(UTKFACE_PATH, file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Crop tengah karena banyak gambar close-up
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.1)
    face = img[margin:h-margin, margin:w-margin]
    try:
        pred_gender, pred_age_cat = predict_gender_age(face)
    except:
        continue

    if pred_gender == true_gender:
        correct_gender += 1
    if pred_age_cat == true_age_cat:
        correct_age += 1

    total += 1
    if total >= 3000:  # Batas agar tidak terlalu lama (ubah sesuai kebutuhan)
        break

# Cetak hasil
print("Total data:", total)
print(f"Akurasi Gender: {correct_gender/total*100:.2f}%")
print(f"Akurasi Age Category: {correct_age/total*100:.2f}%")
