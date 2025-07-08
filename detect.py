import cv2
import math
import argparse
import time

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

# Inisialisasi parser argumen
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to image file (leave empty for webcam)')
args = parser.parse_args()

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

while True:
    # Baca frame
    hasFrame, frame = video.read()
    
    # Jika tidak ada frame (akhir video atau error)
    if not hasFrame:
        if args.image:
            print("End of video file")
        else:
            print("Error reading from camera")
        break
    
    try:
        # Deteksi wajah
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        
        if not faceBoxes:
            print("No face detected")
        else:
            for faceBox in faceBoxes:
                # Ekstrak ROI wajah dengan padding
                face = frame[max(0, faceBox[1]-padding):
                           min(faceBox[3]+padding, frame.shape[0]-1),
                           max(0, faceBox[0]-padding):
                           min(faceBox[2]+padding, frame.shape[1]-1)]
                
                # Prediksi gender
                blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                
                # Prediksi umur
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                
                # Tampilkan hasil
                cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        
        # Tampilkan frame
        cv2.imshow(window_name, resultImg)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except Exception as e:
        print(f"Error during processing: {e}")
        break

# Bersihkan
video.release()
cv2.destroyAllWindows()