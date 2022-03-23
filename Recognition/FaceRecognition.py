from scipy.spatial import distance
import cv2
import tensorflow as tf
import numpy as np
from gtts import gTTS
import os
import gc
import vlc
import time
import imutils
import config
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Load face detection model - MobileFaceNet_SE
interpreter = tf.lite.Interpreter(model_path="Recognition/face_model/mobileFacenet_se.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
inputDetails = interpreter.get_input_details()
outputDetails = interpreter.get_output_details()

# Load face embedding model 
interpreterMask = tf.lite.Interpreter(model_path="Recognition/face_model/mask_detect_mobilenet.tflite")
interpreterMask.allocate_tensors()
# Get input and output tensors.
inputDetailsMask = interpreterMask.get_input_details()
outputDetailsMask = interpreterMask.get_output_details()

# Load masked face embedding model
interpreterMaskEmbed = tf.lite.Interpreter(model_path="Recognition/face_model/mobilefacenet_facex.tflite")
interpreterMaskEmbed.allocate_tensors()
# Get input and output tensors.
inputDetailsMaskEmbed = interpreterMaskEmbed.get_input_details()
outputDetailsMaskEmbed = interpreterMaskEmbed.get_output_details()



class FaceRecognition:

    @staticmethod
    def convertMaskedFaceToArray(face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (112, 112))
        face = face.astype('float32')
        face = (face.transpose((2, 0, 1)) - 127.5)/128.0
        face = np.expand_dims(face, axis=0)
        interpreterMaskEmbed.set_tensor(inputDetailsMaskEmbed[0]['index'], face)
        interpreterMaskEmbed.invoke()
        outputData = interpreterMaskEmbed.get_tensor(outputDetailsMaskEmbed[0]['index'])
        return outputData


    @staticmethod
    def convertFaceToArray(face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (112, 112))
        face = np.expand_dims(face, axis=0)
        face = (face / 255).astype('float32')
        interpreter.set_tensor(inputDetails[0]['index'], face)
        interpreter.invoke()
        outputData = interpreter.get_tensor(outputDetails[0]['index'])
        return outputData

    @staticmethod
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

    @staticmethod
    def faceRecognition(detections, frame, dicts):
        listLabels = []
        listFaces = []

        (h, w) = frame.shape[:2]

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            saveMaxSim = -1
            saveMinDis = 100
            if (confidence < config.DETECTION_CONFIDENCE) or \
                    (detections[0, 0, i, 3:7].max() > 1) or \
                    (detections[0, 0, i, 3] > detections[0, 0, i, 5]) or \
                    (detections[0, 0, i, 4] > detections[0, 0, i, 6]):
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if startX > 0 and startY > 0 and endY > 0 and endX > 0:
                listFaces.append([startX, startY, endX, endY])
                predictLabel = 'người lạ'
                face = frame[startY:endY, startX:endX]

                if face is not None and face != []:
                    checkmask = FaceRecognition.check_mask_bonus(face)
                    if checkmask == 1:
                        faceVector = FaceRecognition.convertFaceToArray(face)
                    if checkmask == 0:
                        faceVector = FaceRecognition.convertMaskedFaceToArray(face)

                    for label in dicts:
                        if label.endswith(str(checkmask)):
                            if checkmask == 1:
                                databaseVector = np.array(dicts.get(label))
                                distances = distance.cdist(faceVector, databaseVector)
                                minDistance = min(np.squeeze(distances))
                                print(minDistance)
                                if minDistance <= config.DISTANCE_NOMASK and minDistance < saveMinDis:
                                    saveMinDis = minDistance
                                    predictLabel = label[:-2]
                                    
                            if checkmask == 0:
                                databaseVector = np.array(dicts.get(label))
                                similarities = distance.cdist(faceVector, databaseVector, metric='cosine')
                                maxSimilarity = max(np.squeeze(similarities))
                                print(maxSimilarity)
                                if maxSimilarity >= DISTANCE_MASK and maxSimilarity > saveMaxSim:
                                    saveMaxSim = maxSimilarity
                                    predictLabel = label[:-2]

                listLabels.append(predictLabel)

        return listLabels, listFaces

    @staticmethod
    def saveUnKownFace(face, local):
        p = os.path.sep.join([local, "{}.jpg".format(str(time.time()))])
        cv2.imwrite(p, face)

    @staticmethod
    def saveFace(face, id, local):
        imageName = str(id) + '_' + str(time.time())
        p = os.path.sep.join([local+id, "{}.jpg".format(imageName)])
        cv2.imwrite(p, face)
        return p

    @staticmethod
    def playSound(name, local):
        p = vlc.MediaPlayer(os.path.join(local, name + ".mp3"))
        p.play()

    @staticmethod
    def classifyFrame(inputQueue, outputQueue, dictQueue):
        net = cv2.dnn.readNetFromCaffe("Recognition/face_model/deploy.prototxt",
                                       "Recognition/face_model/res10_300x300_ssd_iter_140000.caffemodel")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        currentDict = {}
        while True:
            gc.collect()
            while not dictQueue.empty():
                currentDict.update(dictQueue.get())
            if not inputQueue.empty():
                frame = inputQueue.get()
                #frame = imutils.resize(frame, width=640)
                result = {}
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()
                if detections is not None:
                    listLabels, listFaces = FaceRecognition.faceRecognition(detections, frame, currentDict)
                    result['labels'] = listLabels
                    result['faces'] = listFaces
                    result['detections'] = detections
                    result['frame'] = frame
                    outputQueue.put(result)


