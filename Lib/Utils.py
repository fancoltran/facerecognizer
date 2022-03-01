from Recognition.FaceRecognition import FaceRecognition
import config
from Model.AttendanceLog import AttendanceLog
import gc
class Utils:
    @staticmethod
    def removeDictKey(dictionary, key):
        r = dict(dictionary)
        del r[key]
        return r

    @staticmethod
    def gstreamer_pipeline(
        # capture_width=3280,
        # capture_height=2464,
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=29.999999,
        flip_method=6,
        ):
        return(
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )

    @staticmethod
    def saveAttendanceRecord(face, studentId, name):
        path = FaceRecognition.saveFace(face, studentId, config.IMAGE_FOLDER)
        # FaceRecognition.playSound(name, config.SOUND_FOLDER)
        return path

    @staticmethod
    def playSounds(inputName):
        while True:
            gc.collect()
            if not inputName.empty():
                name = inputName.get()
                FaceRecognition.playSound(name, config.SOUND_FOLDER)
                

