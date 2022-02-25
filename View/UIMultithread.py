import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from Recognition.FaceRecognition import FaceRecognition
from Model.Account import Account
from View.FaceListFrame import FaceListFrame
from View.MyVideoCapture import MyVideoCapture
from Model.AttendanceLog import AttendanceLog
import config
import threading


class UI:
    def __init__(self, window, windowTitle, videoSource=0):
        self.window = window
        self.window.title(windowTitle)
        self.videoSource = videoSource
        self.data = UI.getData()

        self.frameToShow = None
        self.frameToPredict = None
        self.photo = None
        self.net = cv2.dnn.readNetFromCaffe("Recognition/face_detector/deploy.prototxt",
                                            "Recognition/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.vid = MyVideoCapture(videoSource)
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()
        self.faceFrame = FaceListFrame(self.window)
        self.canvasFaces = self.faceFrame.canvasFaces
        self.nameLabels = self.faceFrame.nameLabels

        self.faces = []
        self.labels = []
        self.delay = 20
        self.predictThread = threading.Thread(target=self.predict)
        # self.showThread = threading.Thread(target=self.updateFrame)
        self.predictThread.start()
        self.updateFrame()
        self.window.mainloop()

    @staticmethod
    def getData():
        Account.update("")
        dicts = Account.getFaces()
        return dicts

    def updateFrame(self):       
        if self.frameToShow is not None:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.frameToShow))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        for i in range(len(self.faces)):
            self.canvasFaces[i].create_image(90, 90, image=self.faces[i])
            self.nameLabels[i].config(text=str(self.labels[i]))

        self.window.after(self.delay, self.updateFrame)

    def predict(self):
        while True:
            ret, frame = self.vid.getFrame()
            self.frameToPredict = frame
            bboxes = self.detectFace(self.frameToPredict)
            print(len(bboxes))
            listLabels = FaceRecognition.predictLabels(bboxes, self.data, self.frameToPredict)
            self.frameToShow= self.frameToPredict
            for i in range(len(bboxes)):
                box = bboxes[i]
                (startX, startY, endX, endY) = box
                self.frameToShow = cv2.rectangle(self.frameToShow, (startX, startY), (endX, endY), (0, 255, 0), 2)
                face = self.frameToPredict[startY:endY, startX:endX]
                studentId = ''
                if listLabels[i] == config.STRANGER_LABEL:
                    name = config.STRANGER_LABEL
                else:
                    name, studentId = FaceRecognition.getIdName(listLabels[i])

                print(i, name)
                if face.shape[0] > 100 and name != config.STRANGER_LABEL and name not in self.labels:
                    self.faces.append(ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(face, (150, 200)))))
                    self.labels.append(name)
                    UI.saveAttendanceRecord(face, studentId, name)

                if len(self.faces) > config.NUM_FACES:
                    self.faces.pop(0)
                    self.labels.pop(0)
          

    @staticmethod
    def saveAttendanceRecord(face, studentId, name):
        path = FaceRecognition.saveFace(face, studentId, config.IMAGE_FOLDER)
        AttendanceLog.save(studentId, path)
        FaceRecognition.voice(name, config.SOUND_FOLDER)
        AttendanceLog.send()

    def detectFace(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        # Phat hien khuon mat
        self.net.setInput(blob)
        detections = self.net.forward()
        listBbox = []
        # listFaces = []
        # Loop qua cac khuon mat
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Neu conf lon hon threshold
            if confidence > 0.75:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                listBbox.append(box.astype("int"))

        return listBbox

