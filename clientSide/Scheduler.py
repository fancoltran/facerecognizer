import time
from Model.AttendanceLog import AttendanceLog
from Model.Account import Account


class Scheduler:
    def __init__(self, delayTime):
        self.delayTime = delayTime

    def syncData(self, dictQueue):
        while True:
            Account.syncAccount(dictQueue)
            AttendanceLog.syncAttendanceLog()
            time.sleep(self.delayTime)
