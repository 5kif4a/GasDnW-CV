import datetime as dt
import os
from threading import Thread

import cv2
import imutils
import numpy as np
from dotenv import load_dotenv
from imutils.object_detection import non_max_suppression
from requests import get, post, put, patch, delete, options, head

from keyclipwriter import KeyClipWriter

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# classifiers
haarcascades_path = cv2.data.haarcascades
face_cascade_path = haarcascades_path + "haarcascade_frontalface_default.xml"
fire_cascade_path = os.path.join(PROJECT_DIR, "fire_detection.xml")

# images
error_image_path = os.path.join(PROJECT_DIR, "error.jpg")

# environment variables
dotenv_path = os.path.join(PROJECT_DIR, '.env')
load_dotenv(dotenv_path)
CAMERA_ID = os.environ.get("CAMERA_ID")
LOCATION = os.environ.get("LOCATION")
API_BASE_URL = os.environ.get("API_BASE_URL")

# classifiers and HOGDescriptor
face_cascade = cv2.CascadeClassifier(face_cascade_path)
fire_cascade = cv2.CascadeClassifier(fire_cascade_path)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# camera and error image
camera = cv2.VideoCapture(0)
error_image = cv2.imread(error_image_path)
_, error_image_in_bytes = cv2.imencode('.jpg', error_image)

kcw = KeyClipWriter()

# Colors
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
yellow = (0, 255, 255)

is_first_detection = True
last_detection_time = None
fire_exists = False
five_seconds_passed = False

request_methods = {
    'get': get,
    'post': post,
    'put': put,
    'patch': patch,
    'delete': delete,
    'options': options,
    'head': head,
}


def async_request(method, *args, callback=None, timeout=15, **kwargs):
    """Makes request on a different thread, and optionally passes response to a
    `callback` function when request returns.
    """
    method = request_methods[method.lower()]
    if callback:
        def callback_with_args(response, *args, **kwargs):
            callback(response)

        kwargs['hooks'] = {'response': callback_with_args}
    kwargs['timeout'] = timeout
    thread = Thread(target=method, args=args, kwargs=kwargs)
    thread.start()


def generate_image(buffer):
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + bytearray(buffer) + b'\r\n')


def frame_in_rect(objects, frame, title, color):
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in objects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (x, y, w, h) in pick:
        cv2.rectangle(frame, (x, y), (w, h), color, 2)
        cv2.putText(frame, title, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, blue, 2)


def send_log(recognized_objects, filename):
    data = {
        "camera_id": CAMERA_ID,
        "recognized_objects": recognized_objects,
        "filename": filename
    }
    endpoint = f"{API_BASE_URL}/logs"
    try:
        async_request("post", endpoint, json=data)
        print(f"[INFO] - {dt.datetime.now().strftime('%m/%d/%Y, %H:%M:%S')} - Sending log...")
    except Exception as e:
        print(e)


def gen_video():
    while True:
        _, frame = camera.read()
        if frame is None:
            print("No image!")
            break

        frame = imutils.resize(frame, width=min(480, frame.shape[0]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 3)
        fire = fire_cascade.detectMultiScale(gray, 1.05, 3)
        (persons, _) = hog.detectMultiScale(frame,
                                            winStride=(5, 5),
                                            padding=(2, 2),
                                            scale=1.3)

        persons_count = len(persons)
        fire_count = len(fire)

        global is_first_detection, last_detection_time, fire_exists, five_seconds_passed

        if fire_count > 0:
            fire_exists = True
            filename = None
            if not kcw.recording:
                timestamp = dt.datetime.now()
                filename = f"fire-{timestamp.strftime('%m-%d-%Y %H-%M-%S')}.mp4"
                kcw.start(filename, cv2.VideoWriter_fourcc(*"H264"), 30)

            current_detection_time = dt.datetime.now()

            if is_first_detection:
                send_log(recognized_objects="Warning! Fire detected", filename=filename)
                last_detection_time = current_detection_time
                is_first_detection = False

            time_diff = current_detection_time - last_detection_time
            last_detection_time = current_detection_time

            if time_diff.seconds > 10:
                send_log(recognized_objects="Warning! Fire detected", filename=filename)

        if fire_count == 0:
            fire_exists = False

            if not is_first_detection:
                last_fire_extinction_time = dt.datetime.now()
                time_diff = last_fire_extinction_time - last_detection_time

                if time_diff.seconds > 5:
                    five_seconds_passed = True

        frame_in_rect(faces, frame, "face", green)
        frame_in_rect(fire, frame, "fire", red)
        frame_in_rect(persons, frame, "person", yellow)

        cv2.putText(frame, f"CAMERA {CAMERA_ID} - Location: {LOCATION}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red,
                    2)
        cv2.putText(frame, f"Faces: {len(faces)} - Persons: {persons_count} - Fire in the frame: {len(fire) > 0}",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)

        kcw.update(frame)

        if kcw.recording and not fire_exists and five_seconds_passed:
            kcw.finish()
            five_seconds_passed = False

        _, buffer = cv2.imencode('.jpg', frame)

        yield generate_image(buffer)

    yield generate_image(error_image_in_bytes)
