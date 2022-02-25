import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont
from View.FaceListFrame import FaceListFrame
import imutils
import cv2
import time
from Lib.Utils import Utils

print("[INFO] starting video stream...")
cap = cv2.VideoCapture(Utils.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
#time.sleep(2)  # allow the camera sensor to warm up for 2 seconds

# global variables
text = None
frame = None
y = None
fontText = ImageFont.truetype(font='arial.ttf', size=20, encoding='utf-8')


def updateFrame():
    global frame, window, cap, fontText
    frame = cap.read()[1]
    frame = imutils.resize(frame, width=640)
    pilFrame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    pilFrame = pilFrame.resize((1000, 400))
    frame = ImageTk.PhotoImage(pilFrame)  # to ImageTk format
    del pilFrame

    # Update image
    canvas.create_image(500, 0, anchor=tk.N, image=frame)
    # Repeat every 'interval' ms
    window.after(20, updateFrame)

# View components
window = tk.Tk()
window.title("Diem danh")
window.geometry("1024x600")
canvas = tk.Canvas(window, width=1000, height=400)
canvas.pack(side=tk.TOP, expand=True, fill=tk.Y)

faceFrame = FaceListFrame(window)
canvasFaces = faceFrame.canvasFaces
nameLabels = faceFrame.nameLabels
tempLabels = faceFrame.tempLabels

updateFrame()
window.mainloop()

