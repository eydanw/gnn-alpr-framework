from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App General Settings
    APP_NAME: str = "GNN ALPR Framework"
    ENV: str = "development"

    # Neo4j Settings
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"

    # MongoDB Settings
    MONGO_URI: str = "mongodb://localhost:27017"

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # PostgreSQL Settings
    POSTGRES_URI: str = "postgresql://user:pass@localhost:5432/alpr_db"

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Instantiate single global settings object
settings = Settings()