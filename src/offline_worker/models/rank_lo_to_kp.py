"""
Rank existing KP embeddings against one learning object's text.

Lệnh:
    python src/offline_worker/models/rank_lo_to_kp.py --text "Stack and queue basics"
    python src/offline_worker/models/rank_lo_to_kp.py --text-file lo_description.txt
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
EXPECTED_DIMENSIONS = 1024


def rank_kps(
    lo_embedding: np.ndarray,
    kp_records: List[Dict],
    top_k: int,
) -> List[Dict]:
    if not kp_records:
        raise ValueError("Neo4j không có KP nào chứa embedding hợp lệ.")
    if top_k < 1:
        raise ValueError("--top-k phải lớn hơn 0.")

    kp_embeddings = np.asarray(
        [record["embedding"] for record in kp_records],
        dtype=np.float32,
    )
    if kp_embeddings.ndim != 2 or kp_embeddings.shape[1] != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Embedding KP phải có {EXPECTED_DIMENSIONS} chiều; "
            f"nhận được shape {kp_embeddings.shape}."
        )
    if lo_embedding.shape != (EXPECTED_DIMENSIONS,):
        raise ValueError(
            f"Embedding LO phải có shape ({EXPECTED_DIMENSIONS},); "
            f"nhận được {lo_embedding.shape}."
        )

    kp_norms = np.linalg.norm(kp_embeddings, axis=1, keepdims=True)
    if np.any(kp_norms == 0):
        raise ValueError("Phát hiện embedding KP toàn 0; hãy tạo lại embedding.")
    kp_embeddings = kp_embeddings / kp_norms

    lo_norm = np.linalg.norm(lo_embedding)
    if lo_norm == 0:
        raise ValueError("Embedding LO toàn 0, không thể tính cosine similarity.")
    lo_embedding = lo_embedding / lo_norm

    scores = kp_embeddings @ lo_embedding
    order = np.argsort(-scores, kind="stable")[: min(top_k, len(kp_records))]
    return [
        {
            "rank": rank,
            "kp_id": str(kp_records[index]["kp_id"]),
            "title": str(kp_records[index]["title"] or ""),
            "score": float(scores[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tính và in top KP phù hợp nhất với nội dung một LO."
    )
    text_source = parser.add_mutually_exclusive_group(required=True)
    text_source.add_argument(
        "--text",
        help="Nội dung LO, có thể ghép tiêu đề, mô tả và nội dung chính.",
    )
    text_source.add_argument(
        "--text-file",
        type=Path,
        help="Đường dẫn file UTF-8 chứa nội dung LO.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        help="Nếu cung cấp, lưu danh sách đã xếp hạng ra CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    if args.text_file:
        if not args.text_file.is_file():
            raise FileNotFoundError(f"Không tìm thấy file LO: {args.text_file}")
        lo_text = args.text_file.read_text(encoding="utf-8").strip()
    else:
        lo_text = args.text.strip()
    if not lo_text:
        raise ValueError("Nội dung LO không được để trống.")

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")

    print(f"Đang tải mô hình {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    lo_embedding = model.encode(
        lo_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    if lo_embedding.shape != (EXPECTED_DIMENSIONS,):
        raise RuntimeError(
            f"{MODEL_NAME} phải tạo vector {EXPECTED_DIMENSIONS} chiều, "
            f"nhận được {lo_embedding.shape}."
        )

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            kp_records = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (kp:KP)
                    WHERE kp.embedding IS NOT NULL
                      AND size(kp.embedding) = $dimensions
                    RETURN kp.kp_id AS kp_id,
                           coalesce(kp.title, kp.normalized_name, kp.kp_id) AS title,
                           kp.embedding AS embedding
                    ORDER BY kp.kp_id
                    """,
                    dimensions=EXPECTED_DIMENSIONS,
                )
            ]
    finally:
        driver.close()

    results = rank_kps(lo_embedding, kp_records, args.top_k)
    print(f"\nTop {len(results)} KP phù hợp theo cosine similarity:")
    print(f"{'Rank':>4}  {'Score':>7}  {'KP ID':36}  Tên KP")
    for result in results:
        print(
            f"{result['rank']:>4}  {result['score']:>7.4f}  "
            f"{result['kp_id']:36}  {result['title']}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=["rank", "kp_id", "title", "score"],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"\nĐã lưu danh sách xếp hạng: {args.out}")


if __name__ == "__main__":
    main()
