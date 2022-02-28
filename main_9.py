import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from View.FaceListFrame import FaceListFrame
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import imutils
import cv2
from Model.Account import Account
from Model.AttendanceLog import AttendanceLog
from Recognition.FaceRecognition import FaceRecognition
from gtts import gTTS
import time
# edit
from Lib.Utils import Utils
from Lib.Temperature import Temperature
import config
from datetime import datetime
from Scheduler import Scheduler

labels = []
faces = []
temps = []
facesList = []

def getData():
    Account.update()
    Account.updateSound()
    dicts = Account.getFaces("")
    return dicts

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(Utils.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
time.sleep(2)  # allow the camera sensor to warm up for 2 seconds

# share variables between processes
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None
tempQueue = Queue(maxsize=1)
dictQueue = Queue()
dictQueue.put(getData())
# start process
print("[INFO] starting face detection process...")

p0 = Process(target=FaceRecognition().classifyFrame, args=(inputQueue, outputQueue, dictQueue))
p0.daemon = True
p0.start()

print("[INFO] starting thermal detection process...")
p1 = Process(target=Temperature().temp2Que, args=(tempQueue,))
p1.daemon = True
p1.start()
preFaces = []
preTemps = []
preLabels = []
print("[INFO] starting update process...")
p2 = Process(target=Scheduler(3000).syncData, args=(dictQueue,))
p2.daemon = True
p2.start()
fontText = ImageFont.truetype(font='Assets/arial.ttf', size=20, encoding='utf-8')

def updateFrame():
    global frame, window, cap, detections, tImg, break_frame, faces, labels, facesList, preFaces, preTemps, preLabels, fontText
    frame = cap.read()[1]
    frame = imutils.resize(frame, width=640)
    break_frame += 1
    listFaces = []
    listLabels = []
    endFaceBefore, endFaceAfter = [], []
    if facesList:
        endFaceBefore = facesList[-1]
    
    if (break_frame % 1 == 0) and frame is not None:
        break_frame = 0
        if inputQueue.empty(): inputQueue.put(frame)
        
        if not outputQueue.empty():
            result = outputQueue.get()
            detections = result['detections']
            listLabels = result['labels']
            listFaces = result['faces']
            orgFrame = cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB)
        
        if not tempQueue.empty(): tImg = tempQueue.get()
        if detections is not None and tImg is not None:
            listTMaxs = Temperature.calculateTemperature(detections, tImg)
            if listFaces:
                preFaces = listFaces
                preLabels = listLabels
            preTemps = listTMaxs
            for i in range(min(len(listLabels), len(listFaces), len(listTMaxs))):
                label = listLabels[i].split('_')[0]
                face = listFaces[i]
                (sX, sY, eX, eY) = face
                faceImg = orgFrame[sY:eY, sX:eX]
                faceToSave = cv2.cvtColor(cv2.resize(faceImg, (150, 200)), cv2.COLOR_BGR2RGB)
                
                
                if label != 'người lạ' and label not in labels:
                    faceImg = ImageTk.PhotoImage(Image.fromarray(cv2.resize(faceImg, (150, 150))))
                    studentId = listLabels[i].split('_')[1]
                    labels.append(label)
                    faces.append(face)
                    facesList.append(faceImg)
                    path = Utils.saveAttendanceRecord(faceToSave, studentId, label)
                    AttendanceLog.save(studentId, path)
                    temps.append(listTMaxs[i])
            
            if len(labels) > config.NUM_FACES:
                labels.pop(0)
                faces.pop(0)
                facesList.pop(0)
                temps.pop(0)

    # show bounding boxes and labels
    pilFrame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(pilFrame)

    for face, tMax, label in zip(preFaces, preTemps, preLabels):
        name = label.split('_')[0]
        if name != 'người lạ':
            name = name.split(' ')[-1]

        draw.rectangle([(face[0], face[1]), (face[2], face[3])], outline='red')

        xy = (face[0], max(face[1] - 20, 0))
        draw.text(xy, "{} {:.1f}".format(name, tMax), font=fontText, fill='green')

    pilFrame = pilFrame.resize((1000, 400))
    frame = ImageTk.PhotoImage(pilFrame)  # to ImageTk format

    del draw
    del pilFrame

    if facesList:
        endFaceAfter = facesList[-1]
    
    # Update image
    labelImg.configure(image=frame)

    if endFaceAfter != endFaceBefore:
        for i in range(min(len(faces), config.NUM_FACES)):
            canvasFaces[i].configure(image=facesList[i])
            nameLabels[i].config(text=f"{labels[i]}")
            tempLabels[i].config(text=f"{round(temps[i], 1)}°C")
    
    # Repeat every 'interval' ms
    window.after(20, updateFrame)


# global variables
text = None
frame = None
y = None
break_frame = 0
tImg = None

# View components
window = tk.Tk()
window.title("Diem danh")
window.geometry("1024x600")
photo = None

labelImg = tk.Label(window, width=1000, height=400)
labelImg.pack(side=tk.TOP, expand=True, fill=tk.Y)

faceFrame = FaceListFrame(window)

canvasFaces = faceFrame.canvasFaces
nameLabels = faceFrame.nameLabels
tempLabels = faceFrame.tempLabels

delay = 5
updateFrame()

window.mainloop()

