"""
Import CSV exports back into Neo4j Aura.
Nodes are merged on their business IDs (kp_id / learner_id).
Relationships are matched via an in-memory map of internal _id → business ID.
"""

import csv
import os
from neo4j import GraphDatabase

import os

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── helpers ──────────────────────────────────────────────────────────────────

def read_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def coerce(value):
    """Return None for empty strings, otherwise the raw string."""
    return value if value != "" else None


def batch(iterable, size=500):
    it = list(iterable)
    for i in range(0, len(it), size):
        yield it[i : i + size]


# ── node import ───────────────────────────────────────────────────────────────

def import_kp_nodes(session, rows):
    query = """
    UNWIND $rows AS r
    MERGE (n:KP {kp_id: r.kp_id})
    SET n.title            = r.title,
        n.normalized_name  = r.normalized_name,
        n.definition       = r.definition,
        n.domain           = r.domain,
        n.difficulty_label = r.difficulty_label,
        n.difficulty_score = toFloat(r.difficulty_score),
        n.skill_level      = r.skill_level,
        n.source           = r.source,
        n.examples         = r.examples,
        n.semantic_tags    = r.semantic_tags,
        n.created_at       = r.created_at
    """
    for chunk in batch(rows):
        params = [
            {k: coerce(v) for k, v in row.items()}
            for row in chunk
        ]
        session.run(query, rows=params)
    print(f"  KP nodes: {len(rows)} merged")


def import_learner_nodes(session, rows):
    query = """
    UNWIND $rows AS r
    MERGE (n:Learner {learner_id: r.learner_id})
    """
    for chunk in batch(rows):
        params = [{"learner_id": row["learner_id"]} for row in chunk]
        session.run(query, rows=params)
    print(f"  Learner nodes: {len(rows)} merged")


# ── relationship import ───────────────────────────────────────────────────────

def build_id_map(kp_rows, learner_rows):
    """Map internal _id → (label, business_id)."""
    id_map = {}
    for row in kp_rows:
        id_map[row["_id"]] = ("KP", row["kp_id"])
    for row in learner_rows:
        id_map[row["_id"]] = ("Learner", row["learner_id"])
    return id_map


def resolve(id_map, internal_id):
    entry = id_map.get(internal_id)
    if entry is None:
        raise KeyError(f"Unknown node id: {internal_id}")
    return entry


def import_kp_kp_rels(session, rows, id_map, rel_type, extra_props):
    """Generic KP→KP relationship importer."""
    query = f"""
    UNWIND $rows AS r
    MATCH (a:KP {{kp_id: r.src_id}})
    MATCH (b:KP {{kp_id: r.tgt_id}})
    MERGE (a)-[rel:{rel_type}]->(b)
    SET rel += r.props
    """
    resolved = []
    for row in rows:
        _, src_id = resolve(id_map, row["_source_id"])
        _, tgt_id = resolve(id_map, row["_target_id"])
        props = {p: coerce(row[p]) for p in extra_props if p in row}
        if "weight" in props and props["weight"] is not None:
            props["weight"] = float(props["weight"])
        resolved.append({"src_id": src_id, "tgt_id": tgt_id, "props": props})

    for chunk in batch(resolved):
        session.run(query, rows=chunk)
    print(f"  {rel_type}: {len(rows)} merged")


def import_knows_rels(session, rows, id_map):
    query = """
    UNWIND $rows AS r
    MATCH (l:Learner {learner_id: r.learner_id})
    MATCH (k:KP      {kp_id:      r.kp_id})
    MERGE (l)-[rel:KNOWS]->(k)
    SET rel.mastery          = toFloat(r.mastery),
        rel.next_review_date = r.next_review_date,
        rel.updated_at       = r.updated_at
    """
    resolved = []
    for row in rows:
        _, learner_id = resolve(id_map, row["_source_id"])
        _, kp_id      = resolve(id_map, row["_target_id"])
        resolved.append({
            "learner_id":      learner_id,
            "kp_id":           kp_id,
            "mastery":         coerce(row.get("mastery", "0")) or "0",
            "next_review_date": coerce(row.get("next_review_date")),
            "updated_at":      coerce(row.get("updated_at")),
        })

    for chunk in batch(resolved):
        session.run(query, rows=chunk)
    print(f"  KNOWS: {len(rows)} merged")


# ── constraints ───────────────────────────────────────────────────────────────

def ensure_constraints(session):
    session.run("CREATE CONSTRAINT kp_id IF NOT EXISTS FOR (n:KP) REQUIRE n.kp_id IS UNIQUE")
    session.run("CREATE CONSTRAINT learner_id IF NOT EXISTS FOR (n:Learner) REQUIRE n.learner_id IS UNIQUE")
    print("  Constraints ensured")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Reading CSVs…")
    kp_rows       = read_csv("nodes_KP.csv")
    learner_rows  = read_csv("nodes_Learner.csv")
    prereq_rows   = read_csv("rels_IS_PREREQUISITE_OF.csv")
    knows_rows    = read_csv("rels_KNOWS.csv")
    part_rows     = read_csv("rels_PART_OF.csv")
    related_rows  = read_csv("rels_RELATED_TO.csv")
    similar_rows  = read_csv("rels_SIMILAR_TO.csv")

    id_map = build_id_map(kp_rows, learner_rows)

    print("Connecting to Neo4j…")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("Connected.\n")

    with driver.session(database=NEO4J_DATABASE) as session:
        print("── Constraints ──")
        ensure_constraints(session)

        print("\n── Nodes ──")
        import_kp_nodes(session, kp_rows)
        import_learner_nodes(session, learner_rows)

        print("\n── Relationships ──")
        import_kp_kp_rels(session, prereq_rows,  id_map, "IS_PREREQUISITE_OF", ["weight", "updated_at"])
        import_kp_kp_rels(session, part_rows,    id_map, "PART_OF",            ["weight", "updated_at"])
        import_kp_kp_rels(session, related_rows, id_map, "RELATED_TO",         ["weight", "updated_at"])
        import_kp_kp_rels(session, similar_rows, id_map, "SIMILAR_TO",         ["weight", "updated_at"])
        import_knows_rels(session, knows_rows, id_map)

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
