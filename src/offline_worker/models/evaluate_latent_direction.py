"""
Direction / consistency check cho các cạnh tiên quyết ẩn (is_latent=true).
So sánh 3 nhóm cặp (u -> v):
  - GOC   : cạnh tiên quyết gốc do chuyên gia gán (is_latent=false)
  - AN    : cạnh ẩn do GraphSAGE sinh ra (is_latent=true)
  - NGAU_NHIEN: cặp (u, v) ngẫu nhiên (đối chứng)

Chỉ số:
  1. P(difficulty_u <= difficulty_v)  : tiên quyết thường dễ hơn hoặc bằng
  2. P(difficulty_u <  difficulty_v)  : chặt hơn (bỏ trường hợp bằng nhau)
  3. P(bloom_u <= bloom_v)            : remember < understand < ... < create
  4. P(cùng domain)
  5. P(u,v có quan hệ phụ PART_OF/RELATED_TO/SIMILAR_TO theo bất kỳ hướng nào)
  6. Thống kê confidence của cạnh ẩn
Kèm khoảng tin cậy Wilson 95% và p-value (chi-square) AN vs NGAU_NHIEN, AN vs GOC.

Chạy:
    python -m dotenv run -- python src/offline_worker/models/evaluate_latent_direction.py
"""

import sys
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from offline_worker.models.dataset import Neo4jPyGDatasetBuilder, BLOOM_TO_IDX


# ----------------------------------------------------------------------------
# Đọc dữ liệu từ Neo4j
# ----------------------------------------------------------------------------
def load_nodes(session):
    q = """
    MATCH (kp:KP)
    RETURN kp.kp_id AS kp_id,
           kp.difficulty_score AS difficulty,
           coalesce(kp.skill_level, 'remember') AS skill_level,
           kp.domain AS domain
    """
    rows = []
    for r in session.run(q):
        diff = r["difficulty"]
        rows.append(
            {
                "kp_id": str(r["kp_id"]),
                "difficulty": float(diff) if diff is not None else np.nan,
                "bloom": BLOOM_TO_IDX.get(str(r["skill_level"] or "remember").lower(), 0),
                "domain": r["domain"],
            }
        )
    return pd.DataFrame(rows).set_index("kp_id")


def load_prereq_edges(session, latent: bool):
    cond = "= true" if latent else "= false"
    q = f"""
    MATCH (u:KP)-[r:IS_PREREQUISITE_OF]->(v:KP)
    WHERE coalesce(r.is_latent, false) {cond}
    RETURN u.kp_id AS src, v.kp_id AS dst, coalesce(r.confidence, 1.0) AS conf
    """
    return pd.DataFrame(
        [{"src": str(r["src"]), "dst": str(r["dst"]), "conf": float(r["conf"])} for r in session.run(q)]
    )


def load_aux_pairs(session):
    """Tập cặp (u, v) vô hướng có quan hệ PART_OF / RELATED_TO / SIMILAR_TO."""
    q = """
    MATCH (a:KP)-[r:PART_OF|RELATED_TO|SIMILAR_TO]->(b:KP)
    RETURN a.kp_id AS a, b.kp_id AS b
    """
    pairs = set()
    for r in session.run(q):
        a, b = str(r["a"]), str(r["b"])
        pairs.add((a, b))
        pairs.add((b, a))
    return pairs


# ----------------------------------------------------------------------------
# Chỉ số
# ----------------------------------------------------------------------------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centre - half, centre + half)


def compute_metrics(pairs: pd.DataFrame, nodes: pd.DataFrame, aux_pairs):
    """Trả về dict {metric: (k, n)} để tính tỷ lệ + CI + kiểm định."""
    u = nodes.loc[pairs["src"]].reset_index(drop=True)
    v = nodes.loc[pairs["dst"]].reset_index(drop=True)

    mask_d = ~(u["difficulty"].isna() | v["difficulty"].isna())
    du, dv = u.loc[mask_d, "difficulty"].values, v.loc[mask_d, "difficulty"].values

    mask_dom = ~(u["domain"].isna() | v["domain"].isna())
    same_dom = (u.loc[mask_dom, "domain"].values == v.loc[mask_dom, "domain"].values)

    aux_hit = np.array(
        [(s, d) in aux_pairs for s, d in zip(pairs["src"], pairs["dst"])]
    )

    return {
        "diff_u<=v": (int((du <= dv).sum()), int(mask_d.sum())),
        "diff_u<v": (int((du < dv).sum()), int(mask_d.sum())),
        "bloom_u<=v": (int((u["bloom"].values <= v["bloom"].values).sum()), len(pairs)),
        "same_domain": (int(same_dom.sum()), int(mask_dom.sum())),
        "aux_relation": (int(aux_hit.sum()), len(pairs)),
    }


def chi2_pvalue(k1, n1, k2, n2):
    if n1 == 0 or n2 == 0:
        return np.nan
    table = np.array([[k1, n1 - k1], [k2, n2 - k2]])
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return np.nan
    return chi2_contingency(table)[1]


def sample_random_pairs(node_ids, n, rng):
    src = rng.choice(node_ids, size=n)
    dst = rng.choice(node_ids, size=n)
    keep = src != dst
    return pd.DataFrame({"src": src[keep], "dst": dst[keep]})


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="latent_direction_results.csv")
    args = ap.parse_args()

    builder = Neo4jPyGDatasetBuilder()
    with builder.driver.session() as session:
        nodes = load_nodes(session)
        orig = load_prereq_edges(session, latent=False)
        latent = load_prereq_edges(session, latent=True)
        aux = load_aux_pairs(session)
    builder.close()

    print(f"KP: {len(nodes):,} | Cạnh gốc: {len(orig):,} | Cạnh ẩn: {len(latent):,}")
    if len(latent) == 0:
        print("❌ Không có cạnh is_latent=true trong Neo4j.")
        return

    # Cạnh ẩn không được trùng cạnh gốc (kiểm tra thêm)
    orig_set = set(zip(orig["src"], orig["dst"]))
    dup = sum((s, d) in orig_set for s, d in zip(latent["src"], latent["dst"]))
    rev = sum((d, s) in orig_set for s, d in zip(latent["src"], latent["dst"]))
    print(f"Cạnh ẩn trùng cạnh gốc: {dup} | ngược hướng với cạnh gốc: {rev}")

    # Đối chứng: cặp ngẫu nhiên chọn trong các nút xuất hiện ở đồ thị gốc
    rng = np.random.default_rng(args.seed)
    active = pd.unique(pd.concat([orig["src"], orig["dst"]]))
    rand_pairs = sample_random_pairs(active, args.n_random, rng)

    groups = {
        "GOC": compute_metrics(orig, nodes, aux),
        "AN": compute_metrics(latent, nodes, aux),
        "NGAU_NHIEN": compute_metrics(rand_pairs, nodes, aux),
    }

    metric_names = {
        "diff_u<=v": "P(độ khó u ≤ v)",
        "diff_u<v": "P(độ khó u < v)",
        "bloom_u<=v": "P(Bloom u ≤ v)",
        "same_domain": "P(cùng domain)",
        "aux_relation": "P(có PART_OF/RELATED/SIMILAR)",
    }

    rows = []
    for key, label in metric_names.items():
        row = {"Chỉ số": label}
        for g, m in groups.items():
            k, n = m[key]
            lo, hi = wilson_ci(k, n)
            row[g] = f"{k / n:.3f} [{lo:.3f}-{hi:.3f}]" if n else "n/a"
        ka, na = groups["AN"][key]
        kg, ng = groups["GOC"][key]
        kr, nr = groups["NGAU_NHIEN"][key]
        row["p(AN vs NGAU_NHIEN)"] = f"{chi2_pvalue(ka, na, kr, nr):.2e}"
        row["p(AN vs GOC)"] = f"{chi2_pvalue(ka, na, kg, ng):.2e}"
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print("\n" + "=" * 100)
    print("📊 KIỂM TRA TÍNH NHẤT QUÁN CỦA CẠNH ẨN (tỷ lệ [Wilson 95% CI])")
    print("=" * 100)
    print(df.to_string(index=False))

    c = latent["conf"]
    print(
        f"\nConfidence cạnh ẩn: mean={c.mean():.3f} median={c.median():.3f} "
        f"min={c.min():.3f} max={c.max():.3f}"
    )

    # Số nút có difficulty mặc định / thiếu: cảnh báo ties
    tie_rate = (
        nodes.loc[latent["src"], "difficulty"].values
        == nodes.loc[latent["dst"], "difficulty"].values
    ).mean()
    print(f"Tỷ lệ cạnh ẩn có độ khó bằng nhau (ties): {tie_rate:.3f}")

    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n✅ Đã lưu: {args.out}")


if __name__ == "__main__":
    main()