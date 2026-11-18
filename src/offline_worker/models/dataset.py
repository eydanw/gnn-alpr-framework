"""
Module: src/offline_worker/models/dataset.py
Trích xuất Đồ thị Tri thức Neo4j sang PyTorch Geometric Data.
Schema thực tế: kp_id, embedding, skill_level, difficulty_score.
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch_geometric.data import Data
from neo4j import GraphDatabase

BLOOM_LEVELS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
BLOOM_TO_IDX = {level: i for i, level in enumerate(BLOOM_LEVELS)}


class Neo4jPyGDatasetBuilder:
    def __init__(
        self,
        uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user: str = os.getenv("NEO4J_USER", "neo4j"),
        password: str = os.getenv("NEO4J_PASSWORD", "password123"),
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def fetch_nodes(self) -> Tuple[List[str], np.ndarray]:
        query = """
        MATCH (kp:KP)
        RETURN kp.kp_id AS kp_id,
               kp.embedding AS embedding,
               coalesce(kp.skill_level, "remember") AS skill_level,
               coalesce(kp.difficulty_score, 0.5) AS difficulty_score
        ORDER BY kp.kp_id
        """
        kp_ids = []
        feature_list = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                kp_id = str(record["kp_id"])
                emb = record["embedding"]
                skill = str(record["skill_level"] or "remember").lower()
                diff = record["difficulty_score"]

                # 1. 1024d Semantic Embedding
                if emb is not None and len(emb) == 1024:
                    emb_vec = np.array(emb, dtype=np.float32)
                else:
                    emb_vec = np.zeros(1024, dtype=np.float32)

                # 2. 6d Skill Level One-hot
                bloom_vec = np.zeros(6, dtype=np.float32)
                bloom_idx = BLOOM_TO_IDX.get(skill, 0)
                bloom_vec[bloom_idx] = 1.0

                # 3. 1d Difficulty Score
                diff_val = float(diff) if diff is not None else 0.5
                diff_vec = np.array([diff_val], dtype=np.float32)

                # Ghép thành 1031 chiều: [1024 + 6 + 1]
                full_feat = np.concatenate([emb_vec, bloom_vec, diff_vec], axis=0)
                kp_ids.append(kp_id)
                feature_list.append(full_feat)

        x = (
            np.vstack(feature_list)
            if feature_list
            else np.empty((0, 1031), dtype=np.float32)
        )
        return kp_ids, x

    def fetch_edges(self, kp_id_to_idx: Dict[str, int]) -> np.ndarray:
        query = """
        MATCH (u:KP)-[:IS_PREREQUISITE_OF]->(v:KP)
        RETURN u.kp_id AS src_id, v.kp_id AS dst_id
        """
        edge_pairs = []
        with self.driver.session() as session:
            result = session.run(query)
            for record in result:
                src_id = str(record["src_id"])
                dst_id = str(record["dst_id"])
                if src_id in kp_id_to_idx and dst_id in kp_id_to_idx:
                    edge_pairs.append([kp_id_to_idx[src_id], kp_id_to_idx[dst_id]])

        if edge_pairs:
            edge_index = np.array(edge_pairs, dtype=np.int64).T
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)
        return edge_index

    def build_pyg_data(self) -> Tuple[Data, Dict[str, int], Dict[int, str]]:
        print("⏳ Đang nạp danh sách Knowledge Points và ma trận đặc trưng 1031d...")
        kp_ids, x_arr = self.fetch_nodes()
        num_nodes = len(kp_ids)
        print(f"✅ Đã nạp {num_nodes:,} KPs với ma trận đặc trưng: {x_arr.shape}")

        kp_id_to_idx = {kp_id: i for i, kp_id in enumerate(kp_ids)}
        idx_to_kp_id = {i: kp_id for i, kp_id in enumerate(kp_ids)}

        print("⏳ Đang nạp các quan hệ tiên quyết (:IS_PREREQUISITE_OF)...")
        edge_index_arr = self.fetch_edges(kp_id_to_idx)
        num_edges = edge_index_arr.shape[1] if edge_index_arr.size > 0 else 0
        print(f"✅ Đã nạp {num_edges:,} quan hệ tiên quyết hợp lệ.")

        x_tensor = torch.tensor(x_arr, dtype=torch.float32)
        edge_index_tensor = torch.tensor(edge_index_arr, dtype=torch.long)

        data = Data(x=x_tensor, edge_index=edge_index_tensor)
        data.num_nodes = num_nodes
        return data, kp_id_to_idx, idx_to_kp_id