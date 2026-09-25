"""Byte-offset index for entity_id lookups on the large source2/3 TSVs.

Rather than loading rows into memory, this stores {entity_id: byte_offset}
in a small SQLite database (stdlib, disk-backed, indexed B-tree) and reads
the one needed line via file.seek(offset). RAM use during both build and
query stays flat regardless of file size: build streams the file one line
at a time and flushes offsets in small batches, and a query touches only
the handful of B-tree pages SQLite needs for one primary-key lookup -- never
the whole index, let alone the whole file.

train_source1.tsv and train_ground_truth.tsv are not indexed: at ~2.2M rows
each, a plain linear scan already returns in well under a second, so the
extra ~150-250MB of index for each isn't worth it. train_source2/3 (5M+ rows
each) are the ones a scan takes tens of seconds on -- indexing just those two
gets the win for roughly half the disk cost.
"""
import os
import sqlite3

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
INDEX_DB_PATH = os.path.join(PROJECT_ROOT, "work", "row_index.sqlite")

# table_name -> (tsv path relative to project root, id column name)
INDEXED_FILES = {
    "train_source2": ("dataset/train/train_source2.tsv", "entity_id"),
    "train_source3": ("dataset/train/train_source3.tsv", "entity_id"),
}

_BATCH_SIZE = 100_000


def _build_table(conn, table_name, tsv_relpath, id_col):
    """Binary mode throughout: calling .tell() on a text-mode file inside a
    read loop forces TextIOWrapper to disable its read-ahead buffering (it
    can't know the decoded position without re-syncing the decoder), which
    makes line-by-line reads on a multi-GB file dramatically slower. Byte
    offsets are a binary-file concept anyway -- decode only the one line a
    lookup actually needs, in fetch_row_line.
    """
    path = os.path.join(PROJECT_ROOT, tsv_relpath)
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TABLE {table_name} (entity_id TEXT PRIMARY KEY, offset INTEGER NOT NULL)")

    with open(path, "rb") as f:
        header_cols = f.readline().decode("utf-8").rstrip("\n").split("\t")
        id_idx = header_cols.index(id_col)

        batch = []
        n = 0
        offset = f.tell()
        line = f.readline()
        while line:
            entity_id = line.decode("utf-8").rstrip("\n").split("\t")[id_idx]
            batch.append((entity_id, offset))
            if len(batch) >= _BATCH_SIZE:
                conn.executemany(f"INSERT INTO {table_name} VALUES (?, ?)", batch)
                n += len(batch)
                print(f"  {table_name}: indexed {n:,} rows", flush=True)
                batch.clear()
            offset = f.tell()
            line = f.readline()
        if batch:
            conn.executemany(f"INSERT INTO {table_name} VALUES (?, ?)", batch)
            n += len(batch)
    conn.commit()
    print(f"  {table_name}: {n:,} rows indexed")


def build_index(force=False):
    if os.path.exists(INDEX_DB_PATH) and not force:
        print(f"{INDEX_DB_PATH} already exists (pass force=True to rebuild)")
        return
    os.makedirs(os.path.dirname(INDEX_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(INDEX_DB_PATH)
    try:
        for table_name, (tsv_relpath, id_col) in INDEXED_FILES.items():
            print(f"indexing {tsv_relpath} -> {table_name} ...")
            _build_table(conn, table_name, tsv_relpath, id_col)
    finally:
        conn.close()


def fetch_row_line(table_name, entity_id):
    """Return the raw tab-separated line (no trailing newline) for entity_id
    in the given indexed table, or None if not indexed / not found."""
    if not os.path.exists(INDEX_DB_PATH):
        return None
    conn = sqlite3.connect(INDEX_DB_PATH)
    try:
        row = conn.execute(
            f"SELECT offset FROM {table_name} WHERE entity_id = ?", (entity_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    offset = row[0]
    tsv_relpath, _ = INDEXED_FILES[table_name]
    path = os.path.join(PROJECT_ROOT, tsv_relpath)
    with open(path, "rb") as f:
        f.seek(offset)
        return f.readline().decode("utf-8").rstrip("\n")


def has_index() -> bool:
    return os.path.exists(INDEX_DB_PATH)


if __name__ == "__main__":
    build_index()
    print(f"index built at {INDEX_DB_PATH}")
