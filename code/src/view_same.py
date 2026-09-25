import csv
import os
import re
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))

from utils import (
    TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    view_ground_truth,
)

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ANALYSIS_DB_PATH = os.path.join(_PROJECT_ROOT, "work", "gt_analysis.sqlite")

_FIVE_SIX_DIGIT_RE = re.compile(r"\d{5,6}")
_BATCH_SIZE = 100_000


def print_same_entity_group(source1_identifier):
    """Print the Source-1 record together with every Source-2/3 record the
    ground truth says is the same real-world business, as one group."""
    result = view_ground_truth(source1_identifier)
    if result is None:
        print(f"no ground truth row for {source1_identifier!r}")
        return

    rows = []
    if result["source1"]:
        rows.append(("S1", result["source1"]))
    rows += [("S2", row) for row in result["source2"]]
    rows += [("S3", row) for row in result["source3"]]

    print(f"=== {source1_identifier} :: {len(rows)} record(s) considered the same entity ===")
    for source, row in rows:
        print(f"  [{source}] {row['entity_id']:<14} "
              f"{row['business_name']:<40} {row['business_address']:<50} {row['country']}")
    print()


# ---------------------------------------------------------------------------
# Ground-truth analysis (Step 0 EDA): singleton rate, match-count distribution
# split by S2/S3, whether any S2/S3 id is claimed by more than one S1 entity,
# whether true matches always share a country, what share of S2/S3 records
# match nothing, and missing-field rates per country.
#
# Built as a disk-backed SQLite database rather than Python dicts, same as
# code/utils/index_db.py: batched inserts keep Python-side memory flat
# regardless of file size (2.2M-5.3M rows each here), and the actual counting
# (GROUP BY / COUNT DISTINCT / JOIN) runs in SQLite against the on-disk
# tables instead of materializing everything in RAM at once.
# ---------------------------------------------------------------------------

def _build_country_table(conn):
    conn.execute("DROP TABLE IF EXISTS country")
    conn.execute("""
        CREATE TABLE country (
            entity_id TEXT PRIMARY KEY,
            source TEXT,
            country TEXT,
            name_blank INTEGER,
            addr_blank INTEGER,
            addr_has_5_6_digit INTEGER
        )
    """)
    for source_label, path in (("s1", TRAIN_SOURCE1), ("s2", TRAIN_SOURCE2), ("s3", TRAIN_SOURCE3)):
        t0 = time.time()
        n = 0
        batch = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # header
            for entity_id, name, addr, country in reader:
                batch.append((
                    entity_id, source_label, country,
                    1 if not name else 0,
                    1 if not addr else 0,
                    1 if _FIVE_SIX_DIGIT_RE.search(addr) else 0,
                ))
                if len(batch) >= _BATCH_SIZE:
                    conn.executemany("INSERT INTO country VALUES (?,?,?,?,?,?)", batch)
                    n += len(batch)
                    batch.clear()
        if batch:
            conn.executemany("INSERT INTO country VALUES (?,?,?,?,?,?)", batch)
            n += len(batch)
        conn.commit()
        print(f"  country table: {source_label} {n:,} rows in {time.time()-t0:.0f}s", flush=True)


def _build_ground_truth_tables(conn):
    conn.execute("DROP TABLE IF EXISTS gt_summary")
    conn.execute("CREATE TABLE gt_summary (s1_id TEXT PRIMARY KEY, n_match INTEGER, n_s2 INTEGER, n_s3 INTEGER)")
    conn.execute("DROP TABLE IF EXISTS gt_pairs")
    conn.execute("CREATE TABLE gt_pairs (s1_id TEXT, matched_id TEXT)")

    t0 = time.time()
    n_s1 = 0
    n_pairs = 0
    summary_batch = []
    pairs_batch = []
    with open(TRAIN_GROUND_TRUTH, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # header
        for row in reader:
            s1_id = row[0]
            matched_str = row[1] if len(row) > 1 else ""
            ids = [x for x in matched_str.split(",") if x]
            n_s2 = sum(1 for x in ids if x.startswith("S2-"))
            n_s3 = sum(1 for x in ids if x.startswith("S3-"))
            summary_batch.append((s1_id, len(ids), n_s2, n_s3))
            pairs_batch.extend((s1_id, mid) for mid in ids)
            n_s1 += 1
            n_pairs += len(ids)
            if len(summary_batch) >= _BATCH_SIZE:
                conn.executemany("INSERT INTO gt_summary VALUES (?,?,?,?)", summary_batch)
                summary_batch.clear()
            if len(pairs_batch) >= _BATCH_SIZE:
                conn.executemany("INSERT INTO gt_pairs VALUES (?,?)", pairs_batch)
                pairs_batch.clear()
    if summary_batch:
        conn.executemany("INSERT INTO gt_summary VALUES (?,?,?,?)", summary_batch)
    if pairs_batch:
        conn.executemany("INSERT INTO gt_pairs VALUES (?,?)", pairs_batch)
    conn.execute("CREATE INDEX idx_gt_pairs_matched ON gt_pairs(matched_id)")
    conn.commit()
    print(f"  ground truth: {n_s1:,} S1 rows, {n_pairs:,} match pairs in {time.time()-t0:.0f}s", flush=True)


def _build_analysis_db():
    os.makedirs(os.path.dirname(ANALYSIS_DB_PATH), exist_ok=True)
    if os.path.exists(ANALYSIS_DB_PATH):
        os.remove(ANALYSIS_DB_PATH)
    conn = sqlite3.connect(ANALYSIS_DB_PATH)
    try:
        print("building analysis tables (one-time; streams all 4 train files)...")
        _build_country_table(conn)
        _build_ground_truth_tables(conn)
    finally:
        conn.close()


def _print_analysis_report(conn):
    n_s1, n_singleton = conn.execute(
        "SELECT count(*), sum(CASE WHEN n_match = 0 THEN 1 ELSE 0 END) FROM gt_summary"
    ).fetchone()
    print("=" * 70)
    print(f"S1 entities: {n_s1:,}")
    print(f"Singletons (no true match): {n_singleton:,} ({n_singleton/n_s1:.4%})")

    print("\nMatch-count distribution:")
    for n_match, cnt in conn.execute(
        "SELECT n_match, count(*) FROM gt_summary GROUP BY n_match ORDER BY n_match"
    ):
        print(f"  {n_match:>3} matches: {cnt:,}")

    print("\nS2-match-count distribution:")
    for n, cnt in conn.execute("SELECT n_s2, count(*) FROM gt_summary GROUP BY n_s2 ORDER BY n_s2"):
        print(f"  {n:>3} S2 matches: {cnt:,}")
    print("S3-match-count distribution:")
    for n, cnt in conn.execute("SELECT n_s3, count(*) FROM gt_summary GROUP BY n_s3 ORDER BY n_s3"):
        print(f"  {n:>3} S3 matches: {cnt:,}")

    n_distinct = conn.execute("SELECT count(DISTINCT matched_id) FROM gt_pairs").fetchone()[0]
    n_multi_owner = conn.execute("""
        SELECT count(*) FROM (
            SELECT matched_id FROM gt_pairs GROUP BY matched_id HAVING count(DISTINCT s1_id) > 1
        )
    """).fetchone()[0]
    print(f"\nDistinct matched S2/S3 ids: {n_distinct:,}")
    print(f"  ...claimed by more than one S1 entity: {n_multi_owner:,}")

    n_pairs, n_same = conn.execute("""
        SELECT count(*), sum(CASE WHEN c1.country = c2.country THEN 1 ELSE 0 END)
        FROM gt_pairs p
        JOIN country c1 ON c1.entity_id = p.s1_id
        JOIN country c2 ON c2.entity_id = p.matched_id
    """).fetchone()
    print(f"\nTrue (S1, match) pairs checked for country agreement: {n_pairs:,}")
    print(f"  same country: {n_same:,} ({n_same/n_pairs:.4%})")

    print()
    for src in ("s2", "s3"):
        total = conn.execute("SELECT count(*) FROM country WHERE source = ?", (src,)).fetchone()[0]
        matched = conn.execute(
            "SELECT count(DISTINCT matched_id) FROM gt_pairs WHERE matched_id LIKE ?",
            (src.upper() + "-%",),
        ).fetchone()[0]
        print(f"{src.upper()} total: {total:,}  matched: {matched:,} ({matched/total:.4%})")

    print("\nPer-source, per-country breakdown:")
    for source, country, n, n_blank_name, n_blank_addr, n_has_num in conn.execute("""
        SELECT source, country, count(*) AS n, sum(name_blank), sum(addr_blank), sum(addr_has_5_6_digit)
        FROM country GROUP BY source, country ORDER BY source, n DESC
    """):
        print(f"  {source} {country!r:>10}: n={n:>9,}  blank_name={n_blank_name:>7,} ({n_blank_name/n:.2%})  "
              f"blank_addr={n_blank_addr:>8,} ({n_blank_addr/n:.2%})  "
              f"has_5_6digit_num={n_has_num:>9,} ({n_has_num/n:.2%})")
    print("=" * 70)


def analyze_ground_truth(force_rebuild=False):
    """Run the full Step-0 ground-truth analysis, printing a report. Builds
    (or reuses) a disk-backed SQLite database at ANALYSIS_DB_PATH -- pass
    force_rebuild=True if the underlying dataset files have changed."""
    if force_rebuild or not os.path.exists(ANALYSIS_DB_PATH):
        _build_analysis_db()
    conn = sqlite3.connect(ANALYSIS_DB_PATH)
    try:
        _print_analysis_report(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python view_same.py <source1_entity_id> [<source1_entity_id> ...]")
        print("       python view_same.py --analyze [--force-rebuild]")
        sys.exit(1)

    if sys.argv[1] == "--analyze":
        analyze_ground_truth(force_rebuild="--force-rebuild" in sys.argv[2:])
    else:
        for identifier in sys.argv[1:]:
            print_same_entity_group(identifier)
