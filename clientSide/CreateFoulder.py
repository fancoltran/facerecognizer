from Model.Account import Account
from gtts import gTTS
import os

Account.update()
dicts = Account.getFaces("")

for label in dicts:
    # name = label.split("_")[0]
    # output = gTTS("Bạn" + name + "điểm danh thành công", lang="vi", slow=False)
    # output.save(os.path.join('Sounds', name + ".mp3"))
    # print(name)
    id = label.split("_")[1]
    folder_path = './SaveDetectFace/' + id
    os.makedirs(folder_path, exist_ok=True)
    print(id)
