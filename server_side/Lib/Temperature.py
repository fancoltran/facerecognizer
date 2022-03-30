import numpy as np
import board, busio
import adafruit_mlx90640
import gc
from scipy.spatial import distance
import config


class Temperature:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)  # setup I2C
        self.mlx = adafruit_mlx90640.MLX90640(i2c)  # begin MLX90640 with I2C comm
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ  # 16Hz max
        self.mlxShape = (config.TEMP_MATRIX_HEIGHT, config.TEMP_MATRIX_WIDTH)
        self.tdata = np.zeros((config.TEMP_MATRIX_HEIGHT * config.TEMP_MATRIX_WIDTH,))

    @staticmethod
    def calculateTemperature(detections, tImg):
        listTMaxs = []
        resT = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if (confidence < config.DETECTION_CONFIDENCE) or \
                    (detections[0, 0, i, 3:7].max() > 1) or \
                    (detections[0, 0, i, 3] > detections[0, 0, i, 5]) or \
                    (detections[0, 0, i, 4] > detections[0, 0, i, 6]):
                continue
            w, h = 640, 360
            tbox = detections[0, 0, i, 3:7] * np.array([32, 24, 32, 24])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startTX, startTY, endTX, endTY) = tbox.astype("int")
            (startX, startY, endX, endY) = box.astype("int")
            if startTX > 0: startTX -= 1
            if startTY > 0: startTY -= 1
            if endTX < config.TEMP_MATRIX_WIDTH: endTX += 1
            if endTY < config.TEMP_MATRIX_HEIGHT: endTY += 1
            distanceT = distance.euclidean((startX, startY), (endX, endY))
            location = (startY+endX)/2
            tMax = tImg[startTY:endTY, startTX:endTX].max()
            if distanceT > 205:
                if location <= 212:
                    resT = tMax + 2 + 0.4
                if location > 212 and location <= 426:
                    resT = tMax + 2
                if location > 426:
                    resT = tMax + 2 + 0.2

            if distanceT >= 120 and distanceT <= 205:
                if location <= 212:
                    resT = tMax + 3 + 0.2
                if location > 212 and location <= 426:
                    resT = tMax + 2.8
                if location > 426:
                    resT = tMax + 3 + 0.2

            if distanceT <= 120:
                if location <= 212:
                    resT = tMax + 4 + 0.2
                if location > 212 and location <= 426:
                    resT = tMax + 4
                if location > 426:
                    resT = tMax + 4 + 0.2
            resT = resT + 0.3


            """print("toa do :",(startY+endX)/2)
                                                print("khoang cach ", distanceT)
                                                
                                                print("nhiet do: ", resT)
                                                print("nhiet do goc", tMax)"""
            listTMaxs.append(resT)
        return listTMaxs

    def temp2Que(self, tempQueue):
        while True:
            gc.collect()
            self.mlx.getFrame(self.tdata)  # read eMLX temperatures into frame var
            tImg = (np.reshape(self.tdata, self.mlxShape))  # reshape to 24x32 print(t_img.shape) => (24, 32)
            tempQueue.put(tImg)
      



