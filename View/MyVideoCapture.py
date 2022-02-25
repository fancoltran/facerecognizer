import cv2


class MyVideoCapture:

    def __init__(self, video_source):
        self.vid = cv2.VideoCapture(MyVideoCapture.gstreamerPipeline(flipMethod=0), cv2.CAP_GSTREAMER)

        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

    def getFrame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()

            if ret:
                frame = cv2.resize(frame, (640, 480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None

    @staticmethod
    def gstreamerPipeline(
            captureWidth=640,
            captureHeight=640,
            displayWidth=640,
            displayHeight=640,
            frameRate=60,
            flipMethod=0,
    ):
        return (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    captureWidth,
                    captureHeight,
                    frameRate,
                    flipMethod,
                    displayWidth,
                    displayHeight,
                )
        )