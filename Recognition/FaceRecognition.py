
from scipy.spatial import distance
import tensorflow as tf
import numpy as np
import cv2
from gtts import gTTS
import os
import gc
import vlc
import time
import imutils
# Load the TFLite model and allocate tensors.
import config

interpreter = tf.lite.Interpreter(model_path="Recognition/face_detector/mobileFacenet_se.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
inputDetails = interpreter.get_input_details()
outputDetails = interpreter.get_output_details()

interpreterMask = tf.lite.Interpreter(model_path="Recognition/face_detector/facemask.tflite")
interpreterMask.allocate_tensors()
# Get input and output tensors.
inputDetailsMask = interpreterMask.get_input_details()
outputDetailsMask = interpreterMask.get_output_details()


class FaceRecognition:

    @staticmethod
    def convertFaceToArray(face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (112, 112))
        face = np.expand_dims(face, axis=0)
        face = (face / 255).astype('float32')
        input_data = face
        interpreter.set_tensor(inputDetails[0]['index'], input_data)
        interpreter.invoke()
        outputData = interpreter.get_tensor(outputDetails[0]['index'])
        return outputData

    @staticmethod
    def check_mask_bonus(startX, startY, endX, endY, frame):
        face = frame[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0)
        face = (face / 255).astype('float32')
        input_dataMask = face
        interpreterMask.set_tensor(inputDetailsMask[0]['index'], input_dataMask)
        interpreterMask.invoke()
        predictions = interpreterMask.get_tensor(outputDetailsMask[0]['index'])
        lable = np.argmax(predictions)
        return lable

    @staticmethod
    def faceRecognition(detections, frame, dicts):
        listLabels = []
        listFaces = []

        (h, w) = frame.shape[:2]

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            saveMin = 100
            saveMin_mask = 100
            if (confidence < config.DETECTION_CONFIDENCE) or \
                    (detections[0, 0, i, 3:7].max() > 1) or \
                    (detections[0, 0, i, 3] > detections[0, 0, i, 5]) or \
                    (detections[0, 0, i, 4] > detections[0, 0, i, 6]):
                continue
            # predict

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if startX > 0 and startY > 0 and endY > 0 and endX > 0:
                listFaces.append([startX, startY, endX, endY])
                predictLabel = 'người lạ'
                face = frame[startY:endY, startX:endX]

                if face is not None and face != []:
                    checkmask = FaceRecognition.check_mask_bonus(startX, startY, endX, endY, frame)
                    listFaceVector = []
                    faceVector = FaceRecognition.convertFaceToArray(face)
                    faceVector = np.resize(faceVector, (256,))
                    faceVectors = faceVector.reshape(256)
                    listFaceVector.append(faceVectors)
                    listFaceVector = np.asarray(listFaceVector)
                    for label in dicts:

                        numberLabel = dicts.get(label)
                        numberLabel=np.array(numberLabel)
                        distanceFace = (distance.cdist(listFaceVector, numberLabel))

                        minDistanceFace = min(distanceFace.max(axis=0))

                        if minDistanceFace <= config.DISTANCE_MASK and checkmask == 0:
                            if minDistanceFace <= saveMin:
                                saveMin = minDistanceFace
                                predictLabel = label

                        if minDistanceFace <= config.DISTANCE_NOMASK and checkmask == 1:
                            if minDistanceFace <= saveMin_mask:
                                saveMin_mask = minDistanceFace
                                predictLabel = label

                listLabels.append(predictLabel)

        return listLabels, listFaces


    @staticmethod
    def getIdName(label):
        name = label.split("_")[0]
        id_ = label.split("_")[1]
        return name, id_

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
        net = cv2.dnn.readNetFromCaffe("Recognition/face_detector/deploy.prototxt",
                                       "Recognition/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # keep looping
        currentDict = {}
        while True:
            gc.collect()
            while not dictQueue.empty():
                currentDict.update(dictQueue.get())
            # check to see if there is a frame in our input queue
            if not inputQueue.empty():
                # grab the frame from the input queue, resize it, and
                # construct a blob from it
                frame = inputQueue.get()
                #frame = imutils.resize(frame, width=640)
                result = {}
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

                # set the blob as input to our deep learning object
                # detector and obtain the detections
                net.setInput(blob)
                detections = net.forward()
                if detections is not None:
                    listLabels, listFaces = FaceRecognition.faceRecognition(detections, frame, currentDict)
                    # write the detections to the output queue
                    result['labels'] = listLabels
                    result['faces'] = listFaces
                    result['detections'] = detections
                    result['frame'] = frame
                    outputQueue.put(result)



