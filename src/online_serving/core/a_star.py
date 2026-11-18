"""
Module: online_serving/core/a_star.py
Thuật toán tìm đường học tập thích ứng A* (Adaptive Pathfinding).
Tích hợp:
- Khai thác quan hệ tiên quyết ẩn (is_latent: true) từ GraphSAGE.
- Hàm chi phí động theo độ thành thạo (Mastery-aware) & Bloom Taxonomy.
- Tầng đệm Redis Caching tối ưu độ trễ phục vụ (< 50ms cho cached path).
"""

import os
import json
import heapq
import time
from typing import Dict, List, Optional, Any, Tuple
import networkx as nx
from neo4j import GraphDatabase

# Thử kết nối Redis Cache nếu có sẵn
try:
    import redis

    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
    )
    redis_client.ping()
    HAS_REDIS = True
except Exception:
    redis_client = None
    HAS_REDIS = False


class AdaptiveAStarPlanner:
    def __init__(
        self,
        uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user: str = os.getenv("NEO4J_USER", "neo4j"),
        password: str = os.getenv("NEO4J_PASSWORD", "password123"),
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.graph = nx.DiGraph()
        self.node_metadata: Dict[str, dict] = {}
        self.load_graph_from_neo4j()

    def close(self):
        self.driver.close()

    def load_graph_from_neo4j(self):
        """
        Nạp toàn bộ đồ thị tri thức từ Neo4j:
        Bao gồm cả cạnh chính quy và cạnh ngầm {is_latent: true}.
        """
        print("⏳ Đang tải đồ thị tri thức từ Neo4j vào bộ nhớ A*...")
        t0 = time.time()
        self.graph.clear()
        self.node_metadata.clear()

        query_nodes = """
        MATCH (kp:KP)
        RETURN kp.kp_id AS id,
               coalesce(kp.title, kp.normalized_name, kp.kp_id) AS title,
               coalesce(kp.difficulty_score, 0.5) AS difficulty,
               coalesce(kp.skill_level, "remember") AS skill_level
        """

        query_edges = """
        MATCH (u:KP)-[r:IS_PREREQUISITE_OF]->(v:KP)
        RETURN u.kp_id AS src,
               v.kp_id AS dst,
               coalesce(r.is_latent, false) AS is_latent,
               coalesce(r.confidence, 1.0) AS confidence
        """

        with self.driver.session() as session:
            # 1. Đọc Nodes
            for rec in session.run(query_nodes):
                nid = str(rec["id"])
                self.node_metadata[nid] = {
                    "title": rec["title"],
                    "difficulty": float(rec["difficulty"]),
                    "skill_level": rec["skill_level"],
                }
                self.graph.add_node(nid)

            # 2. Đọc Edges
            for rec in session.run(query_edges):
                src = str(rec["src"])
                dst = str(rec["dst"])
                is_latent = bool(rec["is_latent"])
                conf = float(rec["confidence"])
                self.graph.add_edge(
                    src, dst, is_latent=is_latent, confidence=conf
                )

        print(
            f"✅ Đã nạp {self.graph.number_of_nodes():,} nút và "
            f"{self.graph.number_of_edges():,} cạnh vào đồ thị A* "
            f"({time.time() - t0:.2f}s)!"
        )

    def calculate_step_cost(
        self,
        u: str,
        v: str,
        edge_data: dict,
        learner_mastery: Dict[str, float],
    ) -> float:
        """
        Hàm chi phí chuyển trạng thái g(u, v):
        Kết hợp:
        1. Chi phí cơ sở (Base cost).
        2. Độ thành thạo M_v: Nếu người học chưa vững v (mastery thấp), chi phí học tăng.
        3. Cạnh ngầm (is_latent): Thêm trọng số phạt nhẹ nghịch đảo với độ tin cậy (1 - confidence)
           để thuật toán ưu tiên đường chính quy nếu có.
        """
        base_cost = 1.0
        v_meta = self.node_metadata.get(v, {})
        v_diff = v_meta.get("difficulty", 0.5)
        v_mastery = learner_mastery.get(v, 0.0)

        # Khoảng cách kiến thức cần bù đắp (Mastery Gap)
        mastery_penalty = (1.0 - v_mastery) * v_diff

        # Trọng số cạnh ngầm: tin cậy càng thấp phạt càng nhiều
        latent_penalty = 0.0
        if edge_data.get("is_latent", False):
            conf = edge_data.get("confidence", 0.7)
            latent_penalty = 0.4 * (1.0 - conf)

        return base_cost + mastery_penalty + latent_penalty

    def heuristic(self, current: str, goal: str) -> float:
        """
        Hàm Heuristic h(n): Ước lượng khoảng cách đến mục tiêu.
        Nếu có đường đi trong DAG, dùng độ dài đường đi ngắn nhất topo;
        ngược lại trả về 0.0 (Admissible Heuristic).
        """
        try:
            return float(nx.shortest_path_length(self.graph, current, goal))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0

    def find_learning_path(
        self,
        learner_id: str,
        start_kp_id: str,
        target_kp_id: str,
        learner_mastery: Optional[Dict[str, float]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Tìm lộ trình học tập tối ưu bằng A*.
        Tự động kiểm tra và ghi nhận vào Redis Cache.
        """
        t_start = time.time()
        learner_mastery = learner_mastery or {}

        # 1. Kiểm tra Cache
        cache_key = f"learning_path:{learner_id}:{start_kp_id}:{target_kp_id}"
        if use_cache and HAS_REDIS and redis_client:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                res = json.loads(cached_result)
                res["latency_ms"] = round((time.time() - t_start) * 1000, 2)
                res["cached"] = True
                return res

        # 2. Kiểm tra tính hợp lệ của nút
        if start_kp_id not in self.graph or target_kp_id not in self.graph:
            return {
                "success": False,
                "message": "Start hoặc Target KP không tồn tại trong đồ thị.",
                "path": [],
                "latency_ms": round((time.time() - t_start) * 1000, 2),
            }

        # 3. Thuật toán A* Search
        open_set: List[Tuple[float, float, str, List[str]]] = []
        # (f_score, g_score, current_node, current_path)
        h0 = self.heuristic(start_kp_id, target_kp_id)
        heapq.heappush(open_set, (h0, 0.0, start_kp_id, [start_kp_id]))

        g_scores: Dict[str, float] = {start_kp_id: 0.0}
        visited = set()
        best_path: List[str] = []

        while open_set:
            f, g, current, path = heapq.heappop(open_set)

            if current == target_kp_id:
                best_path = path
                break

            if current in visited and g > g_scores.get(current, float("inf")):
                continue
            visited.add(current)

            for neighbor in self.graph.successors(current):
                edge_data = self.graph.get_edge_data(current, neighbor)
                step_cost = self.calculate_step_cost(
                    current, neighbor, edge_data, learner_mastery
                )
                tentative_g = g + step_cost

                if tentative_g < g_scores.get(neighbor, float("inf")):
                    g_scores[neighbor] = tentative_g
                    h = self.heuristic(neighbor, target_kp_id)
                    f_score = tentative_g + h
                    heapq.heappush(
                        open_set,
                        (f_score, tentative_g, neighbor, path + [neighbor]),
                    )

        latency_ms = round((time.time() - t_start) * 1000, 2)

        # 4. Đóng gói kết quả
        if best_path:
            detailed_path = []
            for i, nid in enumerate(best_path):
                meta = self.node_metadata.get(nid, {})
                edge_info = {}
                if i > 0:
                    prev_nid = best_path[i - 1]
                    edata = self.graph.get_edge_data(prev_nid, nid) or {}
                    edge_info = {
                        "is_latent": edata.get("is_latent", False),
                        "confidence": edata.get("confidence", 1.0),
                    }

                detailed_path.append(
                    {
                        "step": i + 1,
                        "kp_id": nid,
                        "title": meta.get("title", nid),
                        "difficulty": meta.get("difficulty", 0.5),
                        "skill_level": meta.get("skill_level", "remember"),
                        "edge_transition": edge_info if i > 0 else None,
                    }
                )

            result_payload = {
                "success": True,
                "learner_id": learner_id,
                "start_kp_id": start_kp_id,
                "target_kp_id": target_kp_id,
                "total_steps": len(best_path),
                "path": detailed_path,
                "latency_ms": latency_ms,
                "cached": False,
            }

            # Lưu vào Redis Cache (TTL = 1 giờ)
            if use_cache and HAS_REDIS and redis_client:
                redis_client.setex(cache_key, 3600, json.dumps(result_payload))

            return result_payload
        else:
            return {
                "success": False,
                "message": (
                    f"Không tìm thấy đường đi từ {start_kp_id} đến"
                    f" {target_kp_id}."
                ),
                "path": [],
                "latency_ms": latency_ms,
                "cached": False,
            }


if __name__ == "__main__":
    # Test nhanh chức năng A* Planner
    planner = AdaptiveAStarPlanner()

    # Thử nghiệm tìm đường giữa 2 nút có liên kết (lấy thử 1 cạnh trong đồ thị)
    edges = list(planner.graph.edges(data=True))
    if edges:
        sample_edge = edges[0]
        src_test, dst_test = sample_edge[0], sample_edge[1]
        print(f"\n🧪 Thử nghiệm A* Planner từ [{src_test}] đến [{dst_test}]...")

        res = planner.find_learning_path(
            learner_id="learner_001",
            start_kp_id=src_test,
            target_kp_id=dst_test,
            learner_mastery={src_test: 0.8, dst_test: 0.1},
        )

        print(
            f"⏱️ Độ trễ lần 1 (Cold): {res['latency_ms']} ms | Cached:"
            f" {res.get('cached')}"
        )
        print(f"🛣️ Số bước: {res.get('total_steps')}")
        for item in res.get("path", []):
            edge_tr = item.get("edge_transition") or {}
            transition = (
                " (via Latent Edge,"
                f" conf={edge_tr['confidence']:.2f})"
                if edge_tr.get("is_latent")
                else ""
            )
            print(
                f" Step {item['step']}: {item['title']} [{item['kp_id']}]{transition}"
            )

        # Thử lại lần 2 để kiểm tra tốc độ Cache
        res2 = planner.find_learning_path(
            learner_id="learner_001",
            start_kp_id=src_test,
            target_kp_id=dst_test,
            learner_mastery={src_test: 0.8, dst_test: 0.1},
        )
        print(
            f"⏱️ Độ trễ lần 2 (Hot/Cache): {res2['latency_ms']} ms | Cached:"
            f" {res2.get('cached')}"
        )

    planner.close()