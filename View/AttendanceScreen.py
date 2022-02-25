import tkinter as tk
import cv2
from PIL import Image, ImageTk
import numpy as np
from Recognition.ThermalCalculator import ThermalCalculator
from Recognition.FaceRecognition import FaceRecognition
from Model.Account import Account
from View.FaceListFrame import FaceListFrame
from View.MyVideoCapture import MyVideoCapture
from Model.AttendanceLog import AttendanceLog
import config

class AttendanceScreen:
    def __init__(self, window, windowTitle, videoSource=0):
        self.window = window
        self.window.title(windowTitle)
        self.videoSource = videoSource
        self.data = AttendanceScreen.getData()
        self.photo = None
        self.net = cv2.dnn.readNetFromCaffe("Recognition/face_detector/deploy.prototxt",
                                            "Recognition/face_detector/res10_300x300_ssd_iter_140000.caffemodel")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)



        self.vid = MyVideoCapture(videoSource)
        self.tempVid = ThermalCalculator()
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()
        self.faceFrame = FaceListFrame(self.window)
        self.canvasFaces = self.faceFrame.canvasFaces
        self.nameLabels = self.faceFrame.nameLabels

        self.faces = []
        self.labels = []
        self.temp = []
        self.delay = 40
        self.updateFrame()
        self.window.mainloop()


    def tempToQueue(s):
        while True:
            mlx.getFrame(tdata)  # read MLX temperatures into frame var
            tImg = (np.reshape(tdata, mlxShape))  # reshape to 24x32 print(t_img.shape) => (24, 32)
            tempQueue.put(tImg)


    def classifyFrame(net, inputQueue, outputQueue):
        # keep looping
        while True:
            # check to see if there is a frame in our input queue
            if not inputQueue.empty():
                # grab the frame from the input queue, resize it, and
                # construct a blob from it
                frame = inputQueue.get()
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

                # set the blob as input to our deep learning object
                # detector and obtain the detections
                net.setInput(blob)
                detections = net.forward()

                # write the detections to the output queue
                outputQueue.put(detections)

    @staticmethod
    def getData():
        Account.update("")
        dicts = Account.getFaces()
        return dicts

    def updateFrame(self):
        ret, frame = self.vid.getFrame()
        bboxes, tBboxes = self.detectFace(frame)
        print(tBboxes)
        listTMax = self.tempVid.calculateTemperature(tBboxes)
        print("tempurature: ", listTMax)
        listLabels = FaceRecognition.predictLabels(bboxes, self.data, frame)
        for i in range(len(bboxes)):
            box = bboxes[i]
            (startX, startY, endX, endY) = box
            frame = cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            face = frame[startY:endY, startX:endX]
            studentId = ''
            if listLabels[i] == config.STRANGER_LABEL:
                name = config.STRANGER_LABEL
            else:
                name, studentId = FaceRecognition.getIdName(listLabels[i])
            if face.shape[0] > 100 and name != config.STRANGER_LABEL and name not in self.labels:
                self.faces.append(ImageTk.PhotoImage(image=Image.fromarray(cv2.resize(face, (150, 200)))))
                self.temp.append(listTMax[i])
                self.labels.append(name)
                path = FaceRecognition.saveFace(face, studentId, config.IMAGE_FOLDER)

                AttendanceLog.save(studentId, path)
                FaceRecognition.voice(name, config.SOUND_FOLDER)
                AttendanceLog.send()

            if len(self.faces) > config.NUM_FACES:
                self.faces.pop(0)
                self.temp.pop(0)
                self.labels.pop(0)

            frame = cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            for i in range(len(self.faces)):
                self.canvasFaces[i].create_image(90, 90, image=self.faces[i])
                self.nameLabels[i].config(text=str(self.labels[i]) + " " + str(self.temp[i]))

        self.window.after(self.delay, self.updateFrame)

    def detectFace(self, frame):

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        # Phat hien khuon mat
        self.net.setInput(blob)
        detections = self.net.forward()
        listBbox = []
        listTFaces = []
        # listFaces = []
        # Loop qua cac khuon mat
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Neu conf lon hon threshold
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                listBbox.append(box.astype("int"))
                tbox = detections[0, 0, i, 3:7] * np.array([32, 24, 32, 24])
                (startTX, startTY, endTX, endTY) = tbox.astype("int")
                listTFaces.append([startTX, startTY, endTX, endTY])
        # print(len(list_bbox), 'faces')
        return listBbox, listTFaces
