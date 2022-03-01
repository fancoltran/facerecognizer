import imutils
from Model.Account import Account
from Model.AttendanceLog import AttendanceLog

Account.update()
Account.updateSound()
AttendanceLog.scheduleSend()
