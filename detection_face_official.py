import json
from numpy import fliplr
import dlib
import os
import tensorflow as tf
import numpy as np
import cv2
import imutils

# Load model ssd nhan dien mat
net = cv2.dnn.readNetFromCaffe("/u01/colombo/dat_project/face_detector/deploy.prototxt",
                               "/u01/colombo/dat_project/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
eye_cascade = cv2.CascadeClassifier("/u01/colombo/dat_project/face_detector/haarcascade_eye.xml")
landmark_detect = dlib.shape_predictor("/u01/colombo/dat_project/face_detector/shape_predictor_68_face_landmarks.dat")
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/u01/colombo/dat_project/face_detector/mobileFacenet_se.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreterMaskEmbed = tf.lite.Interpreter(model_path="/u01/colombo/dat_project/face_detector/mobilefacenet_facex.tflite")
interpreterMaskEmbed.allocate_tensors()
# Get input and output tensors.
inputDetailsMaskEmbed = interpreterMaskEmbed.get_input_details()
outputDetailsMaskEmbed = interpreterMaskEmbed.get_output_details()


class FaceRegistration:

    @staticmethod
    def convertMaskedFaceToArray(face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (112, 112))
        face = face.astype('float32')
        face = (face.transpose((2, 0, 1)) - 127.5) / 128.0
        face = np.expand_dims(face, axis=0)
        interpreterMaskEmbed.set_tensor(inputDetailsMaskEmbed[0]['index'], face)
        interpreterMaskEmbed.invoke()
        outputData = interpreterMaskEmbed.get_tensor(outputDetailsMaskEmbed[0]['index'])
        return outputData

    @staticmethod
    def convert_face_to_array(face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (112, 112))
        face = np.expand_dims(face, axis=0)
        face = (face / 255).astype('float32')
        input_data = face
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        outputData = interpreter.get_tensor(output_details[0]['index'])
        return outputData

    @staticmethod
    def addFaceMaskToFace(landmarks, frame, face, saved, choice2, x1, y1, g, color_gray):
        # Use input () function to capture from user requirements for mask type and mask colour
        choice1 = color_gray
        points = []
        # We are then access the landmark points for the jawline points
        for n in range(x1, y1):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            point = (x, y)
            points.append(point)
            # cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)

        # Coordinates for the additional 3 points for wide, high coverage mask - in sequence
        mask_a = [(landmarks.part(42).x, landmarks.part(15).y),
                  (landmarks.part(27).x, landmarks.part(27).y),
                  (landmarks.part(39).x, landmarks.part(1).y)]

        # Coordinates for the additional point for wide, medium coverage mask - in sequence
        mask_c = [(landmarks.part(29).x, landmarks.part(g).y)]

        # Coordinates for the additional 5 points for wide, low coverage mask (lower nose points) - in sequence
        mask_e = [(landmarks.part(35).x, landmarks.part(35).y),
                  (landmarks.part(34).x, landmarks.part(34).y),
                  (landmarks.part(33).x, landmarks.part(33).y),
                  (landmarks.part(32).x, landmarks.part(32).y),
                  (landmarks.part(31).x, landmarks.part(31).y)]

        face_mask_a = points + mask_a
        face_mask_c = points + mask_c
        face_mask_e = points + mask_e

        face_mask_a = np.array(face_mask_a, dtype=np.int32)
        face_mask_c = np.array(face_mask_c, dtype=np.int32)
        face_mask_e = np.array(face_mask_e, dtype=np.int32)
        mask_type = {1: face_mask_a, 2: face_mask_c, 3: face_mask_e}

        cv2.polylines(frame, [mask_type[choice2]], True, choice1, thickness=2, lineType=cv2.LINE_8)
        cv2.fillPoly(frame, [mask_type[choice2]], choice1, lineType=cv2.LINE_AA)
        p = os.path.sep.join(["/u01/colombo/dat_project/data_all",
                              "{}.png".format(saved)])

        face_vector = FaceRegistration.convertMaskedFaceToArray(face)
        face_vector = face_vector.reshape(512)
        print(face_vector.shape)
        cv2.imwrite(p, face)

        face_vector = np.resize(face_vector, (512,))
        face_vector = "{}".format(list(face_vector))

        return face_vector, p

    @staticmethod
    def checkTiltAngle(eyes):
        angle = 0
        index = 0
        eye_1 = [None, None, None, None]
        eye_2 = [None, None, None, None]
        for (ex, ey, ew, eh) in eyes:
            if index == 0:
                eye_1 = [ex, ey, ew, eh]
            elif index == 1:
                eye_2 = [ex, ey, ew, eh]

            index = index + 1
        if (eye_1[0] is not None) and (eye_2[0] is not None):
            if eye_1[0] < eye_2[0]:
                left_eye = eye_1
                right_eye = eye_2
            else:
                left_eye = eye_2
                right_eye = eye_1
            left_eye_center = (
                int(left_eye[0] + (left_eye[2] / 2)),
                int(left_eye[1] + (left_eye[3] / 2)))

            right_eye_center = (
                int(right_eye[0] + (right_eye[2] / 2)),
                int(right_eye[1] + (right_eye[3] / 2)))

            left_eye_x = left_eye_center[0]
            left_eye_y = left_eye_center[1]
            right_eye_x = right_eye_center[0]
            right_eye_y = right_eye_center[1]

            delta_x = right_eye_x - left_eye_x
            delta_y = right_eye_y - left_eye_y

            angle = np.arctan(delta_y / delta_x)

            angle = (angle * 180) / np.pi
        return angle

    @staticmethod
    def processImage(image, saved, dicts):
        frame = imutils.resize(image, height=image.shape[0])
        # Chuyen tu frame thanh blob

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()
        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > 0.75:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                face_0 = face
                face_1 = face
                face_2 = face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_blur = cv2.Laplacian(gray, cv2.CV_64F).var()
                if face_blur < 1:
                    dicts = {"status": "FAIL",
                             "message": "Ảnh của bạn có khuôn mặt không được rõ( bị che khuất), xin vui lòng chọn ảnh khác"}
                    return dicts, saved
                eyes = eye_cascade.detectMultiScale(gray[startY:endY, startX:endX], 1.1, 4)
                angle = FaceRegistration.checkTiltAngle(eyes)
                if 100 > angle > - 100:
                    p = os.path.sep.join(["/u01/colombo/dat_project/data_all", "{}.png".format(saved)])
                    face_vector = FaceRegistration.convert_face_to_array(face)
                    face_vector = face_vector.reshape(256)
                    print(face_vector.shape)
                    cv2.imwrite(p, face)

                    face_vector = np.resize(face_vector, (256,))
                    face_vector = "{}".format(list(face_vector))
                    dicts["faces"].append(face_vector)
                    dicts["images"].append(p)
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    landmarks = landmark_detect(gray, rect)
                    frame_0 = frame
                    frame_1 = frame
                    frame_2 = frame
                    saved += 1
                    color_gray = (255, 250, 250)
                    faceMask_vector_0, pMask_0 = FaceRegistration.addFaceMaskToFace(landmarks, frame_0, face_0, saved,
                                                                                    1, 2, 15, 29,
                                                                                    color_gray)
                    dicts["faces"].append(faceMask_vector_0)
                    dicts["images"].append(pMask_0)
                    saved += 1
                    color_gray = (255, 0, 0)
                    faceMask_vector_1, pMask_1 = FaceRegistration.addFaceMaskToFace(landmarks, frame_1, face_1, saved,
                                                                                    1, 2, 15, 29,
                                                                                    color_gray)
                    dicts["faces"].append(faceMask_vector_1)
                    dicts["images"].append(pMask_1)
                    saved += 1
                    color_gray = (0, 0, 0)
                    faceMask_vector_2, pMask_2 = FaceRegistration.addFaceMaskToFace(landmarks, frame_2, face_2, saved,
                                                                                    1, 2, 15, 29,
                                                                                    color_gray)
                    dicts["faces"].append(faceMask_vector_2)
                    dicts["images"].append(pMask_2)
                    saved += 1
                    print('----------------------------------------------')
                    return dicts, saved
                else:
                    dicts = {"status": "FAIL",
                             "message": "Ảnh của bạn nghiêng khuôn mặt quá lớn, xin vui lòng chọn ảnh khác"}
                    return dicts, saved
            else:
                if confidence > 0.2:
                    dicts = {"status": "FAIL",
                             "message": "Ảnh của bạn có khuôn mặt không được rõ( bị che khuất), xin vui lòng chọn ảnh khác"}
                    return dicts, saved
                else:
                    dicts = {"status": "FAIL",
                             "message": "Ảnh của bạn không có khuôn mặt, xin vui lòng chọn ảnh khác"}
                    return dicts, saved
        else:
            dicts = {"status": "FAIL", "message": "Ảnh của bạn không có khuôn mặt, xin vui lòng chọn ảnh khác"}
            return dicts, saved

    @staticmethod
    def dataFace(image):
        saved = 0
        dicts = {"status": "SUCCESS", "faces": [], "images": []}

        image_normal = cv2.imread(image)
        dicts, saved = FaceRegistration.processImage(image_normal, saved, dicts)
        rotated_image = fliplr(image_normal)
        dicts, saved = FaceRegistration.processImage(rotated_image, saved, dicts)
        print("result :" + json.dumps(dicts))
