import base64
import os

from gtts import gTTS


class File:
    def __init__(self, fileName):
        self.fileName = fileName

    def toBase64(self):
        with open(self.fileName, "rb") as imageFile:
            return 'data:image/jpeg;base64,' + base64.b64encode(imageFile.read()).decode('utf-8')

    @staticmethod
    def saveSound(name, folder):
        try:
            output = gTTS("f" + name + "điểm danh thành công", lang="vi", slow=False)
            output.save(os.path.join(folder, name + ".mp3"))
        except:
            return None
        return os.path.join(folder, name + ".mp3")
