import redis
from config.settings import settings


class RedisConnector:

    def __init__(self):
        self.client = None

    def connect(self):
        if not self.client:
            self.client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True,
            )
        return self.client

    def close(self):
        if self.client:
            self.client.close()
            self.client = None


redis_db = RedisConnector()