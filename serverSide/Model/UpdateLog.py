from datetime import datetime
from config import *
import sqlalchemy as db


class UpdateLog:
    @staticmethod
    def getLastUpdateTime():
        table = mainDB.table('UpdateLog')
        query = db.select([table.columns.lastUpdateTime]) \
            .order_by(table.columns.lastUpdateTime.desc()).limit(1)

        updateTime = list(mainDB.executeQuery(query)) if mainDB.executeQuery(query) else None
        if updateTime is None:
            return updateTime
        elif updateTime:
            return updateTime[0][0]
        else:
            return ""

    @staticmethod
    def update(time):
        table = mainDB.table("UpdateLog")
        nowTime = datetime.today().strftime('%d/%m/%Y %H:%M:%S')
        if time == '':
            return mainDB.executeQuery(db.insert(table).values({"lastUpdateTime": nowTime}))
        else:
            return mainDB.executeQuery(db.update(table).where(table.columns.lastUpdateTime == time).values(
                {"lastUpdateTime": nowTime}))



