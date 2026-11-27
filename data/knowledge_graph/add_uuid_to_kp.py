import pandas as pd
import uuid
from pathlib import Path

# Đường dẫn file gốc (thay nếu bạn đổi tên)
nodes_path = Path("./nodes.csv")
rels_path = Path("./relationships.csv")

# Đọc file
df_nodes = pd.read_csv(nodes_path)
df_rels = pd.read_csv(rels_path)

# 1️⃣ Tạo UUID cho mỗi node
df_nodes["kp_id"] = [str(uuid.uuid4()) for _ in range(len(df_nodes))]

# 2️⃣ Tạo map từ title → kp_id
id_map = dict(zip(df_nodes["title"], df_nodes["kp_id"]))

# 3️⃣ Gắn UUID vào relationships
df_rels["source_id"] = df_rels["source"].map(id_map)
df_rels["target_id"] = df_rels["target"].map(id_map)

# 4️⃣ Lưu lại file mới
out_nodes = Path("./nodes_final.csv")
out_rels = Path("./relationships_final.csv")

df_nodes.to_csv(out_nodes, index=False, encoding="utf-8")
df_rels.to_csv(out_rels, index=False, encoding="utf-8")

print("✅ Done!")
print(f"Nodes: {out_nodes}")
print(f"Rels:  {out_rels}")
