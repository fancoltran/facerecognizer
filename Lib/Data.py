from sqlalchemy import create_engine
import sqlalchemy as db
from sqlalchemy_utils import database_exists, create_database


class Data:

    def __init__(self, connectionString):
        self.isConnected = False
        self.connection = None
        self.engine = None
        self.metadata = None
        self.connectionString = connectionString

    def connect(self):
        if not self.isConnected:
            self.isConnected = True
            self.engine = create_engine(self.connectionString)
            if not database_exists(self.engine.url):
                create_database(self.engine.url)
            self.connection = self.engine.connect()
            self.metadata = db.MetaData()

    def executeQuery(self, query):
        try:
            self.connect()
            return self.connection.execute(query)
        except:
            return None

    def insert(self, tableName, records):
        try:
            self.connect()
            query = db.insert(self.table(tableName)).values(records)
            return self.connection.execute(query)
        except:
            return None

    def truncate(self, tableName):
        try:
            self.connect()
            query = db.delete(self.table(tableName))
            return self.connection.execute(query)
        except:
            return None

    def table(self, tableName):
        try:
            self.connect()
            return db.Table(tableName, self.metadata, autoload=True, autoload_with=self.engine)
        except:
            return None
