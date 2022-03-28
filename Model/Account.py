from datetime import datetime
from Lib.File import File
from Lib.Utils import Utils
from config import *
from Lib.Request import Request
import sqlalchemy as db
from Model.UpdateLog import UpdateLog


class Account:
    @staticmethod
    def update():
        updateTime = UpdateLog.getLastUpdateTime()
        if updateTime is not None:
            data = Request.call('Extra/FaceRecognition/API/VHV/FaceData/selectAll', {
                "token": currentToken,
                "deviceId": currentDeviceId,
                "updateTime": updateTime
            })

            if data["status"] == "SUCCESS":
                UpdateLog.update(updateTime)
                items = []
                nowTime = datetime.today().strftime('%d/%m/%Y %H:%M:%S')
                for key in data["response"]["items"]:
                    try:
                        item = data["response"]["items"][key]
                        item["faces"] = str(item["faces"])
                        item = Utils.removeDictKey(item, "_id")
                        item = Utils.removeDictKey(item, "id")
                        item = Utils.removeDictKey(item, "image")
                        item.update({"updatedSound": 0})
                        item.update({"lastLoadFaceTime": nowTime})
                        items.append(item)
                    except:
                        continue

                table = mainDB.table("Account")
                if updateTime != "":
                    deletedAccountIds = data["response"]["deletedIds"]
                    if len(deletedAccountIds) != 0:
                        mainDB.executeQuery(db.delete(table).where(table.columns.accountId.in_(deletedAccountIds)))
                    oldAccountIds = list(mainDB.executeQuery(db.select(table.columns.accountId)))
                    oldAccountIds = [e[0] for e in oldAccountIds]

                    for item in items:
                        if item["accountId"] in oldAccountIds:
                            mainDB.executeQuery(
                                db.update(table).values(item).where(table.columns.accountId == item["accountId"]))
                        else:
                            mainDB.insert(table, item)
                    return True
                else:
                    return mainDB.insert(table, items)
            else:
                return False

    @staticmethod
    def getFaces(lastTime):
        table = mainDB.table("Account")
        if lastTime == "":
            data = list(
                mainDB.executeQuery(db.select([table.columns.accountTitle, table.columns.faces, table.columns.accountId])))
        else:
            lastTime = UpdateLog.getLastUpdateTime()
            data = list(
                mainDB.executeQuery(
                    db.select([table.columns.accountTitle, table.columns.faces, table.columns.accountId])
                        .where(table.columns.lastLoadFaceTime >= lastTime)))
        dictionary = {}
        for key, faces, _id in data:
            results = []
            for row in faces.split("],["):
                row = row.replace('[[', '')
                row = row.replace(']]', '')
                row = [float(num) for num in row.split(", ")]
                results.append(row)
            dictionary.update({1: {key + "_" + _id: results[0]}})
            dictionary.update({0: {key + "_" + _id: results[1:]}})

        return dictionary

    @staticmethod
    def updateSound():
        table = mainDB.table("Account")
        data = list(mainDB.executeQuery(
            db.select(table.columns.accountId, table.columns.accountTitle).where(table.columns.updatedSound == 0)))
        items = []
        for row in data:
            if File.saveSound(row[0], row[1], SOUND_FOLDER):
                items.append({"accountId": row[0], "sound": File.saveSound(row[0], row[1], SOUND_FOLDER), "updatedSound": 1})
        for item in items:
            mainDB.executeQuery(
                db.update(table).values(item).where(table.columns.accountId == item["accountId"]))
        return True

    @staticmethod
    def syncAccount(dictQueue):
        Account.update()
        Account.updateSound()
        faces = Account.getFaces("update")
        if faces:
            dictQueue.put(faces)

