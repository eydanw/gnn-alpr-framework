import os
import csv
from neo4j import GraphDatabase

NEO4J_URI = "neo4j+s://defee510.databases.neo4j.io"
NEO4J_USERNAME = "defee510"
NEO4J_PASSWORD = "A0rRVH8r9ytfPVnQLb9JgdBZsMUIvyJYD0uqKwF2Mew"
NEO4J_DATABASE = "defee510"

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def flatten_properties(props: dict) -> dict:
    flat = {}
    for k, v in props.items():
        if isinstance(v, list):
            flat[k] = "|".join(str(i) for i in v)
        else:
            flat[k] = v
    return flat


def export_nodes(session, output_dir: str):
    result = session.run("CALL db.labels() YIELD label RETURN label")
    labels = [r["label"] for r in result]
    print(f"Found labels: {labels}")

    for label in labels:
        rows = session.run(f"MATCH (n:`{label}`) RETURN n")
        records = []
        fieldnames = set()

        for record in rows:
            node = record["n"]
            props = flatten_properties(dict(node.items()))
            props["_id"] = node.element_id
            props["_labels"] = "|".join(node.labels)
            fieldnames.update(props.keys())
            records.append(props)

        if not records:
            print(f"  Label '{label}': no records, skipping")
            continue

        fieldnames = sorted(fieldnames)
        filename = os.path.join(output_dir, f"nodes_{label}.csv")
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        print(f"  Exported {len(records)} nodes of label '{label}' -> {filename}")


def export_relationships(session, output_dir: str):
    result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    rel_types = [r["relationshipType"] for r in result]
    print(f"Found relationship types: {rel_types}")

    for rel_type in rel_types:
        rows = session.run(
            f"MATCH (a)-[r:`{rel_type}`]->(b) "
            f"RETURN r, a.name AS source_name, labels(a) AS source_labels, "
            f"b.name AS target_name, labels(b) AS target_labels, "
            f"elementId(a) AS source_id, elementId(b) AS target_id"
        )
        records = []
        fieldnames = set()

        for record in rows:
            rel = record["r"]
            props = flatten_properties(dict(rel.items()))
            props["_rel_id"] = rel.element_id
            props["_type"] = rel_type
            props["_source_id"] = record["source_id"]
            props["_source_name"] = record["source_name"]
            props["_source_labels"] = "|".join(record["source_labels"] or [])
            props["_target_id"] = record["target_id"]
            props["_target_name"] = record["target_name"]
            props["_target_labels"] = "|".join(record["target_labels"] or [])
            fieldnames.update(props.keys())
            records.append(props)

        if not records:
            print(f"  Rel type '{rel_type}': no records, skipping")
            continue

        fieldnames = sorted(fieldnames)
        filename = os.path.join(output_dir, f"rels_{rel_type}.csv")
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)

        print(f"  Exported {len(records)} rels of type '{rel_type}' -> {filename}")


def main():
    print(f"Connecting to {NEO4J_URI} ...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session(database=NEO4J_DATABASE) as session:
        print("\n=== Exporting Nodes ===")
        export_nodes(session, OUTPUT_DIR)

        print("\n=== Exporting Relationships ===")
        export_relationships(session, OUTPUT_DIR)

    driver.close()
    print("\nDone. All CSV files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
