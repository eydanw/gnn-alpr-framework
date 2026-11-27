"""
Hold-out edge recovery.

Ý tưởng: giấu 20% cạnh tiên quyết gốc (không cho model thấy khi huấn luyện và
message passing), rồi đo model có khôi phục được các cạnh đó hay không.
So sánh với baseline: Random, Common Neighbors, Jaccard, Preferential
Attachment, Cosine(embedding).

Chỉ số: AUC, Average Precision (AP), Hits@10, MRR (xếp hạng cạnh thật giữa
99 đích ngẫu nhiên), lặp nhiều seed -> mean ± std.

Chạy:
    python -m dotenv run -- python src/offline_worker/models/evaluate_holdout.py
    python ... evaluate_holdout.py --epochs 100 --seeds 5 --test-ratio 0.2
"""

import sys
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling

from offline_worker.models.dataset import Neo4jPyGDatasetBuilder
from offline_worker.models.model import GNNPrerequisiteModel

N_RANK_NEG = 99  # số đích âm cho mỗi cạnh dương khi tính Hits@k / MRR


# ----------------------------------------------------------------------------
# Huấn luyện (giữ cùng công thức với train_link_predictor, nhưng chỉ dùng
# cạnh TRAIN để tránh rò rỉ cạnh test)
# ----------------------------------------------------------------------------
def train_on_train_edges(model, x, train_edges, num_nodes, epochs, lr, device):
    model = model.to(device)
    x = x.to(device)
    train_edges = train_edges.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    n = train_edges.size(1)

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        perm = torch.randperm(n, device=device)
        split = int(n * 0.7)
        mp_edges = train_edges[:, perm[:split]]
        pos = train_edges[:, perm[split:]]
        neg = negative_sampling(
            edge_index=train_edges,
            num_nodes=num_nodes,
            num_neg_samples=pos.size(1),
            method="sparse",
        ).to(device)

        z = model.encode(x, mp_edges)
        pairs = torch.cat([pos, neg], dim=1).long()
        labels = torch.cat(
            [torch.ones(pos.size(1), device=device), torch.zeros(neg.size(1), device=device)]
        )
        logits = model.predictor(z[pairs[0]], z[pairs[1]])
        loss = crit(logits, labels)
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs:
            print(f"    epoch {ep:3d}/{epochs} | loss {loss.item():.4f}")
    return model


# ----------------------------------------------------------------------------
# Scorer: hàm (src_idx, dst_idx) -> np.ndarray điểm
# ----------------------------------------------------------------------------
def make_gnn_scorer(model, x, train_edges, device):
    model.eval()
    with torch.no_grad():
        z = model.encode(x.to(device), train_edges.to(device))

    def score(src, dst):
        with torch.no_grad():
            s = torch.as_tensor(src, device=device)
            d = torch.as_tensor(dst, device=device)
            return model.predictor(z[s], z[d]).cpu().numpy()

    return score


def make_structural_scorers(train_edges, num_nodes):
    ei = train_edges.numpy()
    A = sp.coo_matrix(
        (np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(num_nodes, num_nodes)
    ).tocsr()
    A = ((A + A.T) > 0).astype(np.float32).tocsr()  # vô hướng hóa để đếm láng giềng
    deg = np.asarray(A.sum(axis=1)).ravel()

    def common_neighbors(src, dst):
        return np.asarray(A[src].multiply(A[dst]).sum(axis=1)).ravel()

    def jaccard(src, dst):
        cn = common_neighbors(src, dst)
        union = deg[src] + deg[dst] - cn
        return np.where(union > 0, cn / np.maximum(union, 1e-9), 0.0)

    def pref_attach(src, dst):
        return deg[src] * deg[dst]

    return {
        "CommonNeighbors": common_neighbors,
        "Jaccard": jaccard,
        "PrefAttachment": pref_attach,
    }


def make_cosine_scorer(x):
    emb = torch.nn.functional.normalize(x[:, :1024], p=2, dim=1).numpy()

    def cosine(src, dst):
        return (emb[src] * emb[dst]).sum(axis=1)

    return cosine


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def evaluate_scorer(score_fn, test_pos, test_neg, num_nodes, rng):
    # 1) AUC / AP: cạnh thật vs cặp âm (không phải cạnh thật nào)
    s_pos = score_fn(test_pos[0], test_pos[1])
    s_neg = score_fn(test_neg[0], test_neg[1])
    y = np.concatenate([np.ones_like(s_pos), np.zeros_like(s_neg)])
    s = np.concatenate([s_pos, s_neg])
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)

    # 2) Ranking: mỗi cạnh dương xếp hạng giữa 99 đích ngẫu nhiên
    n = test_pos.shape[1]
    neg_dst = rng.integers(0, num_nodes, size=(n, N_RANK_NEG))
    src_rep = np.repeat(test_pos[0], N_RANK_NEG)
    s_cand = score_fn(src_rep, neg_dst.ravel()).reshape(n, N_RANK_NEG)
    gt = (s_cand > s_pos[:, None]).sum(1) + 0.5 * (s_cand == s_pos[:, None]).sum(1)
    rank = 1.0 + gt  # tie được tính trung bình
    return {
        "AUC": auc,
        "AP": ap,
        "Hits@10": float((rank <= 10).mean()),
        "MRR": float((1.0 / rank).mean()),
    }


def split_edges(edge_index, test_ratio, rng):
    n = edge_index.size(1)
    perm = rng.permutation(n)
    n_test = int(n * test_ratio)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    return edge_index[:, train_idx], edge_index[:, test_idx]


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--out", type=str, default="holdout_results.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Thiết bị: {device}")

    builder = Neo4jPyGDatasetBuilder()
    data, _, _ = builder.build_pyg_data(include_latent=False)
    builder.close()

    x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes

    # Cạnh ẩn do GNN sinh ra đã nằm trong Neo4j (is_latent=true) nên sẽ lẫn vào
    # edge_index. Hold-out phải chạy trên đồ thị GỐC, nếu không là tự đánh giá
    # trên cạnh do chính model tạo ra. Cảnh báo để bạn kiểm tra.
    print(
        f"⚠️ edge_index hiện có {edge_index.size(1):,} cạnh. "
        "Nếu > 7,759 nghĩa là đã gồm cạnh ẩn: hãy lọc is_latent=false "
        "(xem --help / ghi chú trong file) trước khi lấy số liệu cuối."
    )

    emb_norm = x[:, :1024].norm(dim=1).mean().item()
    if emb_norm == 0:
        print(
            "⚠️ Embedding 1024d toàn 0 (Neo4j chưa có thuộc tính `embedding`). "
            "Model chỉ còn học từ Bloom + độ khó + cấu trúc; baseline Cosine vô nghĩa. "
            "Nên nạp embedding trước khi chạy số liệu cuối."
        )

    all_rows = []
    for seed in range(args.seeds):
        print(f"\n===== Seed {seed} =====")
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)

        train_e, test_e = split_edges(edge_index, args.test_ratio, rng)
        print(f"Train edges: {train_e.size(1):,} | Test edges: {test_e.size(1):,}")

        # cặp âm test: không trùng bất kỳ cạnh thật nào (train + test)
        test_neg = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=test_e.size(1),
            method="sparse",
        ).numpy()
        test_pos = test_e.numpy()

        model = GNNPrerequisiteModel(1031, 256, 128, 0.2)
        model = train_on_train_edges(
            model, x, train_e, num_nodes, args.epochs, args.lr, device
        )

        scorers = {"GraphSAGE+Bilinear": make_gnn_scorer(model, x, train_e, device)}
        scorers.update(make_structural_scorers(train_e, num_nodes))
        scorers["CosineEmbedding"] = make_cosine_scorer(x)
        scorers["Random"] = lambda s, d, _r=np.random.default_rng(seed + 999): _r.random(len(s))

        for name, fn in scorers.items():
            m = evaluate_scorer(fn, test_pos, test_neg, num_nodes, rng)
            m.update({"method": name, "seed": seed})
            all_rows.append(m)
            print(
                f"  {name:20s} AUC={m['AUC']:.4f} AP={m['AP']:.4f} "
                f"Hits@10={m['Hits@10']:.4f} MRR={m['MRR']:.4f}"
            )

    df = pd.DataFrame(all_rows)
    summary = (
        df.groupby("method")[["AUC", "AP", "Hits@10", "MRR"]]
        .agg(["mean", "std"])
        .round(4)
    )
    order = [
        "GraphSAGE+Bilinear",
        "CosineEmbedding",
        "CommonNeighbors",
        "Jaccard",
        "PrefAttachment",
        "Random",
    ]
    summary = summary.loc[[m for m in order if m in summary.index]]

    print("\n" + "=" * 70)
    print(f"📊 KẾT QUẢ HOLD-OUT ({args.seeds} seeds, test_ratio={args.test_ratio})")
    print("=" * 70)
    print(summary.to_string())

    df.to_csv(args.out, index=False)
    summary.to_csv(args.out.replace(".csv", "_summary.csv"))
    print(f"\n✅ Đã lưu: {args.out} và {args.out.replace('.csv', '_summary.csv')}")


if __name__ == "__main__":
    main()