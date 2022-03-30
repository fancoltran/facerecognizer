import cv2 
from Lib.Utils import Utils
import tensorflow as tf  
import time  
import imutils
import config
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
print("[INFO] loading models")
# face detection model
net = cv2.dnn.readNetFromCaffe("Recognition/face_model/deploy.prototxt",
                               "Recognition/face_model/res10_300x300_ssd_iter_140000.caffemodel")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# masked face detection model
interpreterMask = tf.lite.Interpreter(model_path="Recognition/face_model/git_model.tflite")
interpreterMask.allocate_tensors()
inputDetailsMask = interpreterMask.get_input_details()
outputDetailsMask = interpreterMask.get_output_details()

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(Utils.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
time.sleep(2)  # allow the camera sensor to warm up for 2 seconds

def check_mask_bonus(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    face = face.astype('float32')
    interpreterMask.set_tensor(inputDetailsMask[0]['index'], face)
    interpreterMask.invoke()
    predictions = interpreterMask.get_tensor(outputDetailsMask[0]['index'])
    label = np.argmax(predictions)
    return label

while True:
    frame = cap.read()[1]
    frame = imutils.resize(frame, width=640)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # set the blob as input to our deep learning object
    # detector and obtain the detections
    net.setInput(blob)
    detections = net.forward()

    (h, w) = frame.shape[:2]

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if (confidence < config.DETECTION_CONFIDENCE) or \
                (detections[0, 0, i, 3:7].max() > 1) or \
                (detections[0, 0, i, 3] > detections[0, 0, i, 5]) or \
                (detections[0, 0, i, 4] > detections[0, 0, i, 6]):
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        if startX > 0 and startY > 0 and endY > 0 and endX > 0: 
            face = frame[startY:endY, startX:endX] 
            check_mask = check_mask_bonus(face)    
            color = (0, 255, 0) if check_mask == 0  else (0, 0, 255)                            
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
cap.stop()