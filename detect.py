import cv2
import math
import argparse
import time
import os 

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

def dailyReport(report_data, current_time):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
    time_folder = time.strftime("%m-%Y", time.localtime(current_time))  # contoh: 07-2025

    # Buat path folder tujuan
    report_folder = os.path.join("Reports/daily", time_folder, rak_name)
    os.makedirs(report_folder, exist_ok=True)

    # Nama file report
    filename = f"report_{int(current_time)}.txt"
    filepath = os.path.join(report_folder, filename)

    # Simpan laporan
    with open(filepath, 'w') as f:
        f.write(f"Report Time: {timestamp}\n")
        f.write(f"Rak: {rak_name}\n")
        f.write(f"Total People Detected: {report_data['total']}\n")
        f.write(f"Gender Counts:\n")
        for g, count in report_data['gender'].items():
            f.write(f"  {g}: {count}\n")
        f.write(f"Age Category Counts:\n")
        for cat, count in report_data['age_category'].items():
            f.write(f"  {cat}: {count}\n")

    print(f"Saved report: {filepath}")
    return

import csv

def monthlyReport():
    import csv

    time_folder = time.strftime("%m-%Y", time.localtime(current_time))
    daily_root = os.path.join("Reports/daily", time_folder)
    monthly_folder = os.path.join("Reports/monthly")
    os.makedirs(monthly_folder, exist_ok=True)
    monthly_path = os.path.join(monthly_folder, f"{time_folder}.csv")

    all_reports = []

    # Telusuri setiap rak (subfolder)
    if not os.path.exists(daily_root):
        print(f"Tidak ada folder harian: {daily_root}")
        return

    for rak_folder in os.listdir(daily_root):
        rak_path = os.path.join(daily_root, rak_folder)
        if not os.path.isdir(rak_path):
            continue

        for filename in sorted(os.listdir(rak_path)):
            if filename.endswith(".txt"):
                file_path = os.path.join(rak_path, filename)
                try:
                    data = parse_txt(file_path)
                    all_reports.append(data)
                except Exception as e:
                    print(f"❌ Gagal parse {file_path}: {e}")

    if all_reports:
        with open(monthly_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                'date', 'rak', 'total', 'Male', 'Female', 'kid', 'teen', 'adult', 'elder'
            ])
            writer.writeheader()
            writer.writerows(all_reports)
        print(f"✅ Monthly CSV saved: {monthly_path}")
    else:
        print("⚠️ Tidak ada data yang valid untuk laporan bulanan.")


def parse_txt(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    return {
        'date': lines[0].split(": ")[1].split()[0],
        'rak': lines[1].split(": ")[1].strip(),
        'total': int(lines[2].split(": ")[1]),
        'Male': int(lines[4].split(": ")[1]),
        'Female': int(lines[5].split(": ")[1]),
        'kid': int(lines[7].split(": ")[1]),
        'teen': int(lines[8].split(": ")[1]),
        'adult': int(lines[9].split(": ")[1]),
        'elder': int(lines[10].split(": ")[1]),
    }


# Inisialisasi parser argumen
parser = argparse.ArgumentParser()
parser.add_argument('rak', help='Jenis rak di supermarket (misal: electronics)')
parser.add_argument('--image', help='Path to image file (leave empty for webcam)')
args = parser.parse_args()

rak_name = args.rak

# Pastikan model ada
faceProto = "models/opencv_face_detector.pbtxt"
faceModel = "models/opencv_face_detector_uint8.pb"
ageProto = "models/age_deploy.prototxt"
ageModel = "models/age_net.caffemodel"
genderProto = "models/gender_deploy.prototxt"
genderModel = "models/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load model dengan pengecekan error
try:
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Inisialisasi video capture dengan pengecekan
video_source = args.image if args.image else 0
video = cv2.VideoCapture(video_source)

# Tunggu kamera siap (terutama untuk webcam)
time.sleep(2.0)

if not video.isOpened():
    print(f"Error: Could not open video source {video_source}")
    print("Troubleshooting tips:")
    print("1. For webcam: Make sure no other application is using the camera")
    print("2. For video file: Check if the file exists and is a valid video format")
    print("3. Try changing video source index (e.g., use 1 instead of 0)")
    exit()

padding = 20
window_name = "Detecting age and gender"

# Set ukuran window yang reasonable
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

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

# Data statistik
report_data = {
    'total': 0,
    'gender': {'Male': 0, 'Female': 0},
    'age_category': {'kid': 0, 'teen': 0, 'adult': 0, 'elder': 0}
}

# Timer laporan
last_report_time = time.time()
report_interval = 24*60*60  # dalam detik

last_detection_time = 0
detection_interval = 5 # dalam detik

while True:
    hasFrame, frame = video.read()

    if not hasFrame:
        if args.image:
            print("End of video file")
        else:
            print("Error reading from camera")
        break

    current_time = time.time()
    resultImg = frame.copy()  # default, jika tidak dilakukan deteksi

    # Deteksi hanya setiap 5 detik
    if current_time - last_detection_time >= detection_interval:
        last_detection_time = current_time  # update waktu terakhir deteksi

        try:
            resultImg, faceBoxes = highlightFace(faceNet, frame)

            if not faceBoxes:
                print("No face detected")
            else:
                print(f'{len(faceBoxes)} faces detected')
                for faceBox in faceBoxes:
                    face = frame[max(0, faceBox[1]-padding):
                            min(faceBox[3]+padding, frame.shape[0]-1),
                            max(0, faceBox[0]-padding):
                            min(faceBox[2]+padding, frame.shape[1]-1)]

                    blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]

                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]
                    age_category = age_to_category.get(age, "unknown")

                    print(f"Age range: {age}, Category: {age_category}")
                    
                    # Tambah statistik
                    report_data['total'] += 1
                    report_data['gender'][gender] += 1
                    report_data['age_category'][age_category] += 1


                    cv2.putText(resultImg, f'{gender}, {age_category}', (faceBox[0], faceBox[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
   
        except Exception as e:
            print(f"Error during processing: {e}")
            break
        
        if current_time - last_report_time >= report_interval:
            last_report_time = current_time
            dailyReport(report_data, current_time)
            monthlyReport()
            
            # Reset statistik
            report_data = {
                'total': 0,
                'gender': {'Male': 0, 'Female': 0},
                'age_category': {'kid': 0, 'teen': 0, 'adult': 0, 'elder': 0}
            }


    # Tampilkan frame (baik saat deteksi maupun tidak)
    cv2.imshow(window_name, resultImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
video.release()
cv2.destroyAllWindows()