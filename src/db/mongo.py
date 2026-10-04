from pymongo import MongoClient
from config.settings import settings


class MongoConnector:
    def __init__(self):
        self.client = None

    def connect(self):
        if not self.client:
            self.client = MongoClient(settings.MONGO_URI)
        return self.client

    def close(self):
        if self.client:
            self.client.close()
            self.client = None


mongo_db = MongoConnector()