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

labels = []
faces = []
temps = []
facesList = []


def getData():
    Account.update()
    Account.updateSound()
    dicts = Account.getFaces("")
    return dicts


def syncData(dictQueue, saveDataQueue):
    while True:
        Account.syncAccount(dictQueue)
        AttendanceLog.syncAttendanceLog(saveDataQueue)
        time.sleep(300)


def update():
    if not dictQueue.full():
        dictQueue.put(getUpdateData())
        print(getUpdateData().keys())


def exitHandler():
    p0.terminate()
    p1.terminate()
    p0.join()
    p1.join()
    window.destroy()
    cap.release()


print("[INFO] starting video stream...")
cap = cv2.VideoCapture(Utils.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
time.sleep(2)  # allow the camera sensor to warm up for 2 seconds

# share variables between processes
inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None
tempQueue = Queue(maxsize=1)
dictQueue = Queue()
saveDataQueue = Queue()
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
p2 = Process(target=syncData, args=(dictQueue,saveDataQueue,))
p2.daemon = True
p2.start()


def updateFrame():
    global frame, window, cap, detections, tImg, break_frame, faces, labels, facesList, preFaces, preTemps, preLabels
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
                    faceImg = ImageTk.PhotoImage(Image.fromarray(cv2.resize(faceImg, (150, 200))))
                    studentId = listLabels[i].split('_')[1]
                    labels.append(label)
                    faces.append(face)
                    facesList.append(faceImg)
                    path = Utils.saveAttendanceRecord(faceToSave, studentId, label)
                    nowTime = datetime.today().strftime('%d/%m/%Y %H:%M:%S')
                    saveDataQueue.put({"accountId": studentId, "image": path, "checkIn": nowTime, "isSent": 0})
                    # Utils.saveToDb(studentId, path)
                    temps.append(listTMaxs[i])
            
            if len(labels) > config.NUM_FACES:
                labels.pop(0)
                faces.pop(0)
                facesList.pop(0)
                temps.pop(0)

    # show bounding boxes and labels
    pilFrame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pilFrame)
    fontText = ImageFont.truetype(font='arial.ttf', size=30, encoding='utf-8')
    for face, tMax, label in zip(preFaces, preTemps, preLabels):
        name = label.split('_')[0]
        name = name.split(' ')[-1]

        draw.rectangle([(face[0], face[1]), (face[2], face[3])], outline='red')
        draw.text((face[0], face[1] + 30), "{}{:.1f}".format(name, tMax), font=fontText, fill='green')

    frame = ImageTk.PhotoImage(pilFrame)  # to ImageTk format
    
    if facesList:
        endFaceAfter = facesList[-1]
    
    # Update image
    canvas.create_image(500, 10, anchor=tk.N, image=frame)
    
    if endFaceAfter != endFaceBefore:
        for i in range(min(len(faces), config.NUM_FACES)):
            canvasFaces[i].create_image(90, 90, image=facesList[i])
            nameLabels[i].config(text=f"{labels[i]} {round(temps[i], 1)} C")
    
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
canvas = tk.Canvas(window, width=1000, height=430, highlightthickness=1, highlightbackground="black")
canvas.pack(side=tk.TOP, expand=True, fill=tk.Y)

# updateBtn = tk.Button(window, text="Cap nhat", bg="#3CCEEB", command=update)
# updateBtn.pack(side=tk.RIGHT)
# exitBtn = tk.Button(window, text="Thoat", bg="#3CCEEB", command=exitHandler)
# exitBtn.pack(side=tk.RIGHT)

faceFrame = FaceListFrame(window)

canvasFaces = faceFrame.canvasFaces
nameLabels = faceFrame.nameLabels

delay = 5
updateFrame()

window.mainloop()

