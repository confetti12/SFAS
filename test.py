import cv2
import numpy as np
import argparse
import warnings
import time
import os

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

FACE_RECOGNIZER_MODEL = "images/Wajah/training.xml"

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces


def test_image_from_frame(image, model_test, image_cropper, model_dir):
    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0

    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (255, 0, 0)
    else:
        result_text = "FakeFace Score: {:.2f}".format(value/2)
        color = (0, 0, 255)

    return result_text, color

def detect_video(model_dir, device_id):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        result_text, color = test_image_from_frame(frame, model_test, image_cropper, model_dir)
        
        cv2.putText(
            frame,
            result_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        
       
        cv2.imshow('Face Recognition and Spoofing Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")

    args = parser.parse_args()

    detect_video(args.model_dir, args.device_id)
