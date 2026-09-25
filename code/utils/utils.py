import csv
import os

import index_db

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
database_path = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "dataset"))

TRAIN_SOURCE1 = os.path.join(database_path, "train", "train_source1.tsv")
TRAIN_SOURCE2 = os.path.join(database_path, "train", "train_source2.tsv")
TRAIN_SOURCE3 = os.path.join(database_path, "train", "train_source3.tsv")
TRAIN_GROUND_TRUTH = os.path.join(database_path, "train", "train_ground_truth.tsv")

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# path -> index_db table name, for files small enough NOT to bother indexing
# (train_source1/ground_truth) this is simply absent and we always scan.
_INDEX_TABLE_FOR_PATH = {
    TRAIN_SOURCE2: "train_source2",
    TRAIN_SOURCE3: "train_source3",
}


def _row_from_line(line):
    return dict(zip(SOURCE_COLUMNS, line.split("\t")))


def _find_row_by_id(path, entity_id):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["entity_id"] == entity_id:
                return row
    return None


def _find_rows_by_ids(path, wanted_ids):
    wanted = set(wanted_ids)
    found = {}
    if not wanted:
        return found

    index_table = _INDEX_TABLE_FOR_PATH.get(path)
    if index_table and index_db.has_index():
        for entity_id in wanted:
            line = index_db.fetch_row_line(index_table, entity_id)
            if line is not None:
                found[entity_id] = _row_from_line(line)
        return found

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["entity_id"] in wanted:
                found[row["entity_id"]] = row
                if len(found) == len(wanted):
                    break
    return found


def _find_ground_truth_row(source1_identifier):
    with open(TRAIN_GROUND_TRUTH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["source1_entity_id"] == source1_identifier:
                return row
    return None


def view_ground_truth(source1_identifier):
    """Look up a Source-1 entity_id in train_ground_truth.tsv, then fetch its
    matched Source-2 and Source-3 rows (and the Source-1 row itself) from the
    train dataset files.

    Returns a dict:
        {
            "source1": {...} | None,
            "matched_entity_ids": [...],
            "source2": [{...}, ...],
            "source3": [{...}, ...],
        }
    Rows are dicts with keys entity_id/business_name/business_address/country.
    Returns None if source1_identifier has no row in the ground truth file.
    """
    gt_row = _find_ground_truth_row(source1_identifier)
    if gt_row is None:
        return None

    matched_ids = [x for x in gt_row["matched_entity_ids"].split(",") if x]
    s2_ids = [x for x in matched_ids if x.startswith("S2-")]
    s3_ids = [x for x in matched_ids if x.startswith("S3-")]

    source1_row = _find_row_by_id(TRAIN_SOURCE1, source1_identifier)
    source2_rows = _find_rows_by_ids(TRAIN_SOURCE2, s2_ids)
    source3_rows = _find_rows_by_ids(TRAIN_SOURCE3, s3_ids)

    return {
        "source1": source1_row,
        "matched_entity_ids": matched_ids,
        "source2": [source2_rows[i] for i in s2_ids if i in source2_rows],
        "source3": [source3_rows[i] for i in s3_ids if i in source3_rows],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python utils.py <source1_entity_id>")
        sys.exit(1)

    result = view_ground_truth(sys.argv[1])
    if result is None:
        print(f"no ground truth row for {sys.argv[1]!r}")
        sys.exit(1)

    print("source1:", result["source1"])
    print(f"\nmatched_entity_ids ({len(result['matched_entity_ids'])}):", result["matched_entity_ids"])
    print(f"\nsource2 rows ({len(result['source2'])}):")
    for row in result["source2"]:
        print(" ", row)
    print(f"\nsource3 rows ({len(result['source3'])}):")
    for row in result["source3"]:
        print(" ", row)
