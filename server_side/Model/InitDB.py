import sqlalchemy as db
from config import *


class InitDB:
    @staticmethod
    def createTables():
        mainDB.connect()
        db.Table('Account', mainDB.metadata,
                 db.Column('lastUpdateTime', db.TEXT),
                 db.Column('accountId', db.TEXT),
                 db.Column('accountTitle', db.TEXT),
                 db.Column('status', db.TEXT),
                 db.Column('faces', db.TEXT),
                 db.Column('updatedSound', db.INTEGER),
                 db.Column('sound', db.TEXT),
                 db.Column('lastLoadFaceTime', db.TEXT)
                 )
        db.Table('AttendanceLog', mainDB.metadata,
                 db.Column('checkIn', db.TEXT),
                 db.Column('checkOut', db.TEXT),
                 db.Column('accountId', db.TEXT),
                 db.Column('accountTitle', db.TEXT),
                 db.Column('image', db.TEXT),
                 db.Column('isSent', db.INTEGER)
                 )
        db.Table('UpdateLog', mainDB.metadata,
                 db.Column('lastUpdateTime', db.TEXT)
                 )
        mainDB.metadata.create_all(mainDB.engine)



