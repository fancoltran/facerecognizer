import tkinter as tk
from tkinter import ttk
import config
import tkinter.font as tkFont
from PIL import Image, ImageTk
import cv2
class FaceListFrame(ttk.Frame):
    def __init__(self, container):
        super().__init__(container)

        self.rowconfigure(0, weight=2)
        canvasFaces = [None] * config.NUM_FACES
        nameLabels = [None] * config.NUM_FACES
        tempLabels = [None] * config.NUM_FACES
        nameFontStyle = tkFont.Font(family="Lucida Grande", size=10)
        tempFontStyle = tkFont.Font(family="Lucida Grande", size=10)
        for i in range(config.NUM_FACES):
            canvasFaces[i] = tk.Label(self)
            nameLabels[i] = ttk.Label(self, text="Demo", font=nameFontStyle)
            tempLabels[i] = ttk.Label(self, text="", font=tempFontStyle)

        self.canvasFaces = canvasFaces
        self.nameLabels = nameLabels
        self.tempLabels = tempLabels

        self.pack(side=tk.BOTTOM, fill=tk.Y)
        for i in range(config.NUM_FACES):
            tempLabels[i].grid(column=i, row=0)
            canvasFaces[i].grid(column=i, row=1, padx=2, pady=2)
            nameLabels[i].grid(column=i, row=2)

        defaultImage = cv2.imread('Assets/avatar.png')
        pilImage = Image.fromarray(cv2.cvtColor(defaultImage, cv2.COLOR_BGR2RGB))
        pilImage = pilImage.resize((150, 150))
        image = ImageTk.PhotoImage(pilImage)

        for i in range(config.NUM_FACES):
            canvasFaces[i].configure(image=image)