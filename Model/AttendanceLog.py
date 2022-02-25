from config import *
from Lib.File import File
from Lib.Request import Request
import sqlalchemy as db
from datetime import datetime
import json


class AttendanceLog:
    @staticmethod
    def save(accountId, imageUrl):
        table = mainDB.table("AttendanceLog")
        nowTime = datetime.today().strftime('%d/%m/%Y %H:%M:%S')
        return mainDB.insert(table, {"accountId": accountId, "image": imageUrl, "checkIn": nowTime, "isSent": 0})

    @staticmethod
    def syncAttendanceLog():
        table = mainDB.table('AttendanceLog')
        try:
            query = db.select([table.columns.accountId, table.columns.checkIn, table.columns.image]) \
            .where(table.columns.isSent == 0)
        except:
            return None
        items = []
        sentIds = []
        results = mainDB.executeQuery(query)
        for result in results:
            items.append({
                "accountId": result[0],
                "checkInTime": result[1],
                "image": File(result[2]).toBase64()
            })
            sentIds.append(result[0])

        if items:
            req = Request.call('LMS/Attendance/API/VHV/Attendance/multiCheckIn', {
                "token": currentToken,
                "deviceId": currentDeviceId,
                "items": json.dumps(items)
            })
            if req["status"] == "FAIL":
                return None
            else:
                mainDB.executeQuery(
                    db.update(table).values({"isSent": 1}).where(table.columns.accountId.in_(sentIds)))
                return req
        else:
            return None
