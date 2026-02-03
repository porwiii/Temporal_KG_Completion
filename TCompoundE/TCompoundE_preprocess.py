import os
from pathlib import Path
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================

CSV_PATH = "tkgl-smallpedia_edgelist.csv"

DATA_PATH = Path("data_full")
DATASET_NAME = "SMALLPEDIA_FULL"
OUT_DIR = DATA_PATH / DATASET_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATA
# =========================

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)
assert set(df.columns) == {"ts", "head", "tail", "relation_type"}
print(f"Loaded {len(df)} triples")


# =========================
# BUILD STRING-LEVEL SETS
# (as in ICEWS preprocessing)
# =========================

print("Collecting entities / relations / timestamps...")

entities = set(df["head"]).union(set(df["tail"]))
relations = set(df["relation_type"])
timestamps = set(df["ts"].astype(str))

entities_to_id = {e: i for i, e in enumerate(sorted(entities))}
relations_to_id = {r: i for i, r in enumerate(sorted(relations))}
timestamps_to_id = {t: i for i, t in enumerate(sorted(timestamps))}

n_entities = len(entities_to_id)
n_relations = len(relations_to_id)

print(f"{n_entities} entities, {n_relations} relations, {len(timestamps_to_id)} timestamps")

# save mappings (exactly like ICEWS)
for mapping, name in zip(
    [entities_to_id, relations_to_id, timestamps_to_id],
    ["ent_id", "rel_id", "ts_id"]
):
    with open(OUT_DIR / name, "w") as f:
        for k, v in mapping.items():
            f.write(f"{k}\t{v}\n")

# =========================
# SPLIT INTO TRAIN / VALID / TEST
# =========================

train, valid, test = [], [], []

print("Splitting triples...")

for row in df.itertuples(index=False):
    h = entities_to_id[row.head]
    r = relations_to_id[row.relation_type]
    t = entities_to_id[row.tail]
    ts_str = str(row.ts)
    ts = timestamps_to_id[ts_str]

    quad = [h, r, t, ts]
    year = int(row.ts)

    if year >= 2008:
        test.append(quad)
    elif year >= 1998:
        valid.append(quad)
    else:
        train.append(quad)

train_entities = set()
for h, _, t, _ in train:
    train_entities.add(h)
    train_entities.add(t)

def filter_by_train_entities(data):
    return [q for q in data if q[0] in train_entities and q[2] in train_entities]

#valid = filter_by_train_entities(valid)
#test  = filter_by_train_entities(test)


print(f"Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}")
#assert len(train) + len(valid) + len(test) == len(df)

# 1. collect used entities
used_entities = set()
for split in [train, valid, test]:
    for h, _, t, _ in split:
        used_entities.add(h)
        used_entities.add(t)

used_entities = sorted(used_entities)
ent_remap = {old: new for new, old in enumerate(used_entities)}

# 2. remap triples
def remap(data):
    return [
        [ent_remap[h], r, ent_remap[t], ts]
        for h, r, t, ts in data
    ]

train = remap(train)
valid = remap(valid)
test  = remap(test)

n_entities = len(ent_remap)

# sanity check
all_entities = set()
for split in [train, valid, test]:
    for h, _, t, _ in split:
        all_entities.add(h)
        all_entities.add(t)

print("Max entity id after filtering:", max(all_entities))
print("Num entities used:", len(all_entities))

# =========================
# SAVE PICKLES (CORE INPUT)
# =========================

for split, data in zip(
    ["train", "valid", "test"],
    [train, valid, test]
):
    with open(OUT_DIR / f"{split}.pickle", "wb") as f:
        pickle.dump(np.array(data, dtype=np.uint64), f)

# =========================
# CREATE FILTERING LISTS
# =========================

print("Creating filtering lists (to_skip)...")

to_skip = {"lhs": defaultdict(set), "rhs": defaultdict(set)}

#R = 283
R = len(relations)

for split in ["train", "valid", "test"]:
    examples = pickle.load(open(OUT_DIR / f"{split}.pickle", "rb"))
    for lhs, rel, rhs, ts in examples:
        to_skip["lhs"][(rhs, rel + R, ts)].add(lhs)
        to_skip["rhs"][(lhs, rel, ts)].add(rhs)

to_skip_final = {
    side: {k: sorted(v) for k, v in d.items()}
    for side, d in to_skip.items()
}

with open(OUT_DIR / "to_skip.pickle", "wb") as f:
    pickle.dump(to_skip_final, f)

# =========================
# CREATE PROBAS (NEG SAMPLING)
# =========================

print("Computing entity frequencies (probas)...")

examples = pickle.load(open(OUT_DIR / "train.pickle", "rb"))
counters = {
    "lhs": np.zeros(n_entities),
    "rhs": np.zeros(n_entities),
    "both": np.zeros(n_entities),
}

for lhs, _, rhs, _ in examples:
    counters["lhs"][lhs] += 1
    counters["rhs"][rhs] += 1
    counters["both"][lhs] += 1
    counters["both"][rhs] += 1

for k in counters:
    counters[k] /= np.sum(counters[k])

with open(OUT_DIR / "probas.pickle", "wb") as f:
    pickle.dump(counters, f)

print("Done. Dataset ready for TCompoundE.")
