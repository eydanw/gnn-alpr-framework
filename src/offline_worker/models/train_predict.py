"""
Module: src/offline_worker/models/train_predict.py
Huấn luyện GraphSAGE + Bilinear Link Predictor & Khai phá quan hệ tiên quyết ẩn.
Ghi nhận trực tiếp vào Neo4j với thuộc tính {is_latent: true, confidence: prob}.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import time
from typing import Dict, List, Set, Tuple
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

from offline_worker.models.dataset import Neo4jPyGDatasetBuilder
from offline_worker.models.model import GNNPrerequisiteModel


def train_link_predictor(
    model: GNNPrerequisiteModel,
    data,
    epochs: int = 20,
    lr: float = 0.005,
    device: str = "cpu",
) -> GNNPrerequisiteModel:
    model = model.to(device)
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    num_edges = edge_index.size(1)
    print(f"🚀 Bắt đầu huấn luyện GraphSAGE Link Predictor ({epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        # Tách 70% cạnh cho Message Passing, 30% cạnh để học giám sát
        perm = torch.randperm(num_edges)
        split = int(num_edges * 0.7)
        mp_edges = edge_index[:, perm[:split]]
        pos_eval = edge_index[:, perm[split:]]

        neg_eval = negative_sampling(
            edge_index=edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_eval.size(1),
            method="sparse",
        ).to(device)

        z = model.encode(x, mp_edges)
        total_eval = torch.cat([pos_eval, neg_eval], dim=1).long()
        labels = torch.cat(
            [
                torch.ones(pos_eval.size(1), device=device),
                torch.zeros(neg_eval.size(1), device=device),
            ]
        )

        src = total_eval[0]
        dst = total_eval[1]
        logits = model.predictor(z[src], z[dst])
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == epochs:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                acc = (preds == labels).float().mean().item() * 100.0
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | Loss: {loss.item():.4f} | "
                f"Acc: {acc:.2f}% | Time: {(time.time() - t0):.2f}s"
            )

    print("✅ Hoàn tất huấn luyện!")
    return model


def mine_candidate_pairs(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    num_nodes: int,
    num_samples: int = 3500,
    top_k: int = 15,
) -> Tuple[List[int], List[int]]:
    print("\n⏳ Đang khai phá Top-K láng giềng ngữ nghĩa tiềm năng...")
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    existing_edges = set(zip(src_list, dst_list))

    sem_emb = x[:, :1024]
    norm_emb = torch.nn.functional.normalize(sem_emb, p=2, dim=1)

    unique_nodes = list(set(src_list + dst_list))
    if len(unique_nodes) > num_samples:
        np.random.seed(42)
        sample_nodes = list(
            np.random.choice(unique_nodes, size=num_samples, replace=False)
        )
    else:
        sample_nodes = unique_nodes

    sample_t = torch.tensor(sample_nodes, dtype=torch.long)
    sub_emb = norm_emb[sample_t]

    candidate_set: Set[Tuple[int, int]] = set()
    batch_size = 500

    for b in range(0, len(sample_nodes), batch_size):
        b_nodes = sample_nodes[b : b + batch_size]
        b_sub = sub_emb[b : b + batch_size]

        sim = torch.matmul(b_sub, norm_emb.t())
        _, top_idx = torch.topk(sim, k=top_k + 10, dim=1)

        for i, u in enumerate(b_nodes):
            for j in range(1, top_k + 10):
                v = top_idx[i, j].item()
                if u != v and (u, v) not in existing_edges:
                    candidate_set.add((u, v))
                    if len(candidate_set) >= 30000:
                        break
            if len(candidate_set) >= 30000:
                break
        if len(candidate_set) >= 30000:
            break

    print(f"🎯 Đã khai phá được {len(candidate_set):,} cặp ứng viên ngữ nghĩa!")
    cand_src = [pair[0] for pair in candidate_set]
    cand_dst = [pair[1] for pair in candidate_set]
    return cand_src, cand_dst


def predict_and_enrich_neo4j(
    model: GNNPrerequisiteModel,
    data,
    idx_to_kp_id: Dict[int, str],
    threshold: float = 0.60,
    device: str = "cpu",
):
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    print("⏳ Đang tính toán vector nhúng topo z_u qua 2 tầng GraphSAGE...")
    with torch.no_grad():
        z = model.encode(x, edge_index)

    # 1. Trích xuất ứng viên
    cand_src, cand_dst = mine_candidate_pairs(edge_index, x, data.num_nodes)
    cand_src_t = torch.tensor(cand_src, dtype=torch.long, device=device)
    cand_dst_t = torch.tensor(cand_dst, dtype=torch.long, device=device)

    # 2. Chấm điểm xác suất
    print("⏳ Đang chấm điểm xác suất P(u -> v)...")
    with torch.no_grad():
        probs = (
            model.predictor.predict_proba(z[cand_src_t], z[cand_dst_t])
            .cpu()
            .numpy()
        )

    print(
        f"📊 Phân phối xác suất: Min = {probs.min():.4f} | "
        f"Mean = {probs.mean():.4f} | Max = {probs.max():.4f}"
    )

    cutoff = max(threshold, float(np.percentile(probs, 80)))
    high_conf_indices = np.where(probs >= cutoff)[0]
    print(f"🔍 Tìm thấy {len(high_conf_indices):,} cặp có độ tin cậy cao (P >= {cutoff:.3f}).")

    # 3. Lọc khử chu trình DAG (FR-02)
    print("⏳ Đang kiểm tra khử chu trình DAG qua NetworkX...")
    dag = nx.DiGraph()
    dag.add_nodes_from(range(data.num_nodes))
    dag.add_edges_from(edge_index.t().tolist())

    valid_latent_edges = []
    sorted_order = high_conf_indices[np.argsort(-probs[high_conf_indices])]

    for idx in sorted_order:
        u = cand_src[idx]
        v = cand_dst[idx]
        prob = float(probs[idx])

        if not nx.has_path(dag, v, u):
            dag.add_edge(u, v)
            valid_latent_edges.append(
                {"src": idx_to_kp_id[u], "dst": idx_to_kp_id[v], "prob": prob}
            )
        if len(valid_latent_edges) >= 2000:
            break

    print(f"✅ Sau khi lọc DAG: Giữ lại {len(valid_latent_edges):,} quan hệ tiên quyết ẩn hợp lệ (0% Cycle)!")

    # 4. Ghi nhận vào Neo4j (FR-01: {is_latent: true})
    print("⏳ Đang cập nhật các quan hệ ẩn vào cơ sở dữ liệu Neo4j theo kp_id...")
    builder = Neo4jPyGDatasetBuilder()
    cypher_merge = """
    UNWIND $batch AS item
    MATCH (u:KP {kp_id: item.src}), (v:KP {kp_id: item.dst})
    MERGE (u)-[r:IS_PREREQUISITE_OF]->(v)
    SET r.is_latent = true,
        r.confidence = item.prob,
        r.created_by = 'GraphSAGE_Bilinear',
        r.updated_at = datetime()
    """

    batch_size = 500
    total_written = 0
    with builder.driver.session() as session:
        for i in range(0, len(valid_latent_edges), batch_size):
            batch = valid_latent_edges[i : i + batch_size]
            session.run(cypher_merge, batch=batch)
            total_written += len(batch)
            print(f" -> Đã ghi {total_written:,} / {len(valid_latent_edges):,} cạnh...")

    builder.close()
    print(
        f"\n🎉 THÀNH CÔNG RỰC RỠ! Đã làm giàu {len(valid_latent_edges):,} cạnh ẩn "
        f"(:IS_PREREQUISITE_OF {{is_latent: true}}) vào cơ sở dữ liệu Neo4j!"
    )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Thiết bị tính toán: {device}")

    builder = Neo4jPyGDatasetBuilder()
    data, kp_id_to_idx, idx_to_kp_id = builder.build_pyg_data()
    builder.close()

    model = GNNPrerequisiteModel(
        node_in_dim=1031,
        sage_hidden=256,
        node_out_dim=128,
        dropout=0.2,
    )

    trained_model = train_link_predictor(
        model, data, epochs=20, lr=0.005, device=device
    )

    predict_and_enrich_neo4j(
        trained_model, data, idx_to_kp_id, threshold=0.60, device=device
    )