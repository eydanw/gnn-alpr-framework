from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from config.settings import settings


class Base(DeclarativeBase):
    pass


class PostgresConnector:

    def __init__(self):
        self.engine = None
        self.SessionLocal = None

    def connect(self):
        if not self.engine:
            self.engine = create_engine(settings.POSTGRES_URI)
            self.SessionLocal = sessionmaker(
                autocommit=False, autoflush=False, bind=self.engine
            )
        return self.engine


postgres_db = PostgresConnector()