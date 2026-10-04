from neo4j import GraphDatabase
from config.settings import settings


class Neo4jConnector:

    def __init__(self):
        self.driver = None

    def connect(self):
        if not self.driver:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
        return self.driver

    def close(self):
        if self.driver:
            self.driver.close()
            self.driver = None


# Khởi tạo instance kết nối dùng chung
neo4j_db = Neo4jConnector()