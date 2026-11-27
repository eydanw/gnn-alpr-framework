"""
Sinh embedding ngữ nghĩa 1024 chiều bằng BGE-M3 cho các node KP trong Neo4j.
"""

import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


def build_text(record) -> str:
    parts = [
        record["title"],
        record["definition"],
        record["domain"],
        record["semantic_tags"],
        record["examples"],
    ]
    return "\n".join(str(part).strip() for part in parts if part and str(part).strip())


def main():
    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")

    print("Đang tải mô hình BGE-M3...")
    model = SentenceTransformer("BAAI/bge-m3")

    print("Đang đọc Knowledge Points...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        records = list(session.run(
            """
            MATCH (kp:KP)
            RETURN kp.kp_id AS kp_id,
                   coalesce(kp.title, '') AS title,
                   coalesce(kp.definition, '') AS definition,
                   coalesce(kp.domain, '') AS domain,
                   coalesce(kp.semantic_tags, '') AS semantic_tags,
                   coalesce(kp.examples, '') AS examples
            ORDER BY kp.kp_id
            """
        ))

    print(f"Đã đọc {len(records):,} KP.")
    texts = [build_text(record) for record in records]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    if embeddings.shape[1] != 1024:
        driver.close()
        raise RuntimeError(
            f"BGE-M3 phải tạo vector 1024 chiều, nhận được {embeddings.shape[1]}"
        )

    print("Đang ghi embedding vào Neo4j...")
    batch_size = 500
    total = 0
    with driver.session() as session:
        for start in range(0, len(records), batch_size):
            batch = []
            for record, embedding in zip(
                records[start:start + batch_size],
                embeddings[start:start + batch_size],
            ):
                batch.append({
                    "kp_id": str(record["kp_id"]),
                    "embedding": embedding.astype("float32").tolist(),
                })

            session.run(
                """
                UNWIND $batch AS item
                MATCH (kp:KP {kp_id: item.kp_id})
                SET kp.embedding = item.embedding
                """,
                batch=batch,
            )
            total += len(batch)
            print(f" -> Đã ghi {total:,} / {len(records):,} KP")

    driver.close()
    print("Hoàn tất. Neo4j hiện có thuộc tính kp.embedding 1024 chiều.")


if __name__ == "__main__":
    main()
