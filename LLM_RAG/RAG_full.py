# This is almost the same code as in TKGC_RAG_clean notebook,
# but it's hard-coded for hybrid model of threshold=1

import torch
import os, json, time
from pathlib import Path
from transformers import set_seed
import numpy as np
import pandas as pd
import re
from typing import List, Tuple
from collections import defaultdict

from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)

SEED = 42
set_seed(SEED)

# LLM: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
llm_name = "Qwen/Qwen3-4B-Instruct-2507"

# We want to use 4bit quantization to save memory (in case some of you use their own computer)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=False, load_in_4bit=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_name, padding_side="left")

# Prevent some transformers specific issues.
tokenizer.use_default_system_prompt = False
tokenizer.pad_token_id = tokenizer.eos_token_id

# Load LLM.
llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    quantization_config=quantization_config,
    device_map={"": 0}, # load all the model layers on GPU 0
    torch_dtype=torch.bfloat16, # float precision
)

# Set LLM on eval mode.
llm.eval()


qid_to_label = json.load(open("../qid_to_label.json"))
pid_to_label = json.load(open("../pid_to_label.json"))
norm_label_to_qids = json.load(open("../norm_label_to_qids.json"))

# load and split dataset
df = pd.read_csv("../tkgl-smallpedia_edgelist.csv")
df["ts"] = df["ts"].astype(int)

train_df = df[df["ts"] < 2008].copy()
test_df  = df[df["ts"] >= 2008].copy()

print(len(train_df), len(test_df))


def build_hr_index(df):
    idx = defaultdict(list)
    for _, row in df.iterrows():
        h = str(row["head"])
        r = str(row["relation_type"])
        t = str(row["tail"])
        ts = int(row["ts"])
        idx[(h, r)].append((ts, t))
    # sort by ts so it's easy to retreive close entries
    for k in idx:
        idx[k].sort(key=lambda x: x[0])
    return idx

hr_index = build_hr_index(train_df)

def build_head_index(df):
    idx = defaultdict(list)
    for _, row in df.iterrows():
        h = str(row["head"])
        r = str(row["relation_type"])
        t = str(row["tail"])
        ts = int(row["ts"])
        idx[h].append((ts, r, t))
    # sort by ts so it's easy to retreive close entries
    for k in idx:
        idx[k].sort(key=lambda x: x[0])
    return idx

head_index = build_head_index(train_df)

def build_entity_index(df):
    idx = defaultdict(list)
    for _, row in df.iterrows():
        h = str(row["head"])
        r = str(row["relation_type"])
        t = str(row["tail"])
        ts = int(row["ts"])

        idx[h].append((ts, h, r, t))
        idx[t].append((ts, h, r, t))

    for e in idx:
        idx[e].sort(key=lambda x: x[0])
    return idx

entity_index = build_entity_index(train_df)

def build_rel_index(df):
    idx = defaultdict(list)
    for _, row in df.iterrows():
        ts = int(row["ts"])
        h = str(row["head"])
        r = str(row["relation_type"])
        t = str(row["tail"])
        idx[r].append((ts, h, t))
    for r in idx:
        idx[r].sort(key=lambda x: x[0])
    return idx

rel_index = build_rel_index(train_df)

def retrieve_facts_new(head_id, rel_id, ts, k=12):
    out = []

    # 1) exact match: (head, rel) -> (ts, h, r, t)
    facts_hr = hr_index.get((head_id, rel_id), [])  # list[(ts, tail_id)]
    if facts_hr:
        ranked = sorted(facts_hr, key=lambda x: abs(x[0] - ts))
        out.extend([(y, head_id, rel_id, tail_id) for (y, tail_id) in ranked])

    # 2) fallback: (head, *) -> already (ts, r, t) OR (ts, r, tail_id)
    if len(out) < k:
        facts_h = head_index.get(head_id, [])  # list[(ts, r, tail_id)]
        if facts_h:
            ranked_h = sorted(facts_h, key=lambda x: abs(x[0] - ts))
            out.extend([(y, head_id, r, tail_id) for (y, r, tail_id) in ranked_h])

    # 3) fallback: facts about entity (as head OR tail)
    if len(out) < k:
        facts_e = entity_index.get(head_id, [])  # list[(ts, h, r, t)]
        if facts_e:
            ranked_e = sorted(facts_e, key=lambda x: abs(x[0] - ts))
            # dodaj tylko tyle ile brakuje
            need = k - len(out)
            out.extend(ranked_e[:need])

    # 4) relation-only examples (global)
    if len(out) < k:
        facts_r = rel_index.get(rel_id, [])  # [(ts, h, t)]
        ranked_r = sorted(facts_r, key=lambda x: abs(x[0] - ts))
        need = k - len(out)
        out.extend([(y, h, rel_id, t) for (y, h, t) in ranked_r[:need]])
        
    # filter to k
    out = out[:k]

    # change to text
    facts_txt = []
    for y, h, r, t in out:
        h_label = qid_to_label.get(h, h)
        r_label = pid_to_label.get(r, r)
        t_label = qid_to_label.get(t, t)
        facts_txt.append(f"In {y}, {h_label} {r_label} {t_label}.")

    facts_txt = list(dict.fromkeys(facts_txt))  # delete duplicates
    return facts_txt[:k]


def normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\.$", "", s)
    return s

def tail_label_to_qids(tail_label: str):
    n = normalize_label(tail_label)
    return norm_label_to_qids.get(n, [])

# Define the function to call the qwen model
def qwen_llm_candidates(prompt):
    # 1. Instruction prompt
    system_prompt = (
        "You solve Temporal Knowledge Graph Completion (TKGC).\n"
        "You will be given a query (head, relation, timestamp), retrieved facts, and a candidate list.\n"
        "Your task is to select the most likely tail entity from the candidate list.\n\n"
        "Rules:\n"
        "1) Choose up to 3 candidates ONLY from the provided candidate list.\n"
        "2) Order them from best to worst.\n"
        "3) Return ONLY valid JSON in ONE line, and nothing else.\n"
        '4) Output format: {"tail_labels": ["...", "...", "..."]}\n'
        '   If no candidate matches, return {"tail_labels": []}.\n'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {'role': 'user', 'content': prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
        ).to(llm.device)

    with torch.inference_mode():
        outputs = llm.generate(**inputs, max_new_tokens=64, do_sample=False)
        
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return raw

# Define the function to call the qwen model
def qwen_llm_open_top3(prompt):
    
    system_prompt = (
        "You solve Temporal Knowledge Graph Completion (open-world).\n"
        "Given a query (head, relation, timestamp) and retrieved facts, predict the most likely tail entity.\n"
        "The correct tail may NOT appear in the retrieved facts.\n"
        "Return up to 3 tail entity labels ordered best→worst.\n"
        "Return ONLY one-line valid JSON and nothing else.\n"
        'Format: {"tail_labels": ["...","...","..."]}\n'
        'If you cannot propose any, return {"tail_labels": []}.'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {'role': 'user', 'content': prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
        ).to(llm.device)
    
    with torch.inference_mode():
        outputs = llm.generate(**inputs, max_new_tokens=64, do_sample=False)
        
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return raw

def build_user_prompt_candidates_top3(head_label, rel_label, ts, retrieved_facts, candidates):
    facts_block = "\n".join([f"- {f}" for f in retrieved_facts])
    cand_block = "\n".join([f"{i+1}. {c}" for i, c in enumerate(candidates)])

    return (
        f"Query:\nTime: {ts}\nHead: {head_label}\nRelation: {rel_label}\n\n"
        f"Retrieved facts:\n{facts_block}\n\n"
        f"Candidates (pick up to 3):\n{cand_block}\n\n"
        "Rules:\n"
        "1) Choose up to 3 candidates ONLY from the candidate list.\n"
        "2) Order them best→worst.\n"
        '3) Return ONLY JSON: {"tail_labels": ["...","...","..."]}.\n'
        '4) If none match, return {"tail_labels": []}.'
    )


def rag_answer_candidates_top3(head_id: str, rel_id: str, ts: int, k=12):
    head_label = qid_to_label.get(head_id, head_id)
    rel_label  = pid_to_label.get(rel_id, rel_id)

    retrieved_facts = retrieve_facts_new(head_id, rel_id, ts, k=k)
    if not retrieved_facts:
        return [], [], '{"tail_labels": []}'

    candidates = extract_tail_candidates(retrieved_facts, head_label, rel_label)
    if not candidates:
        return [], retrieved_facts, '{"tail_labels": []}'

    user_prompt = build_user_prompt_candidates_top3(head_label, rel_label, ts, retrieved_facts, candidates)
    raw = qwen_llm_candidates(user_prompt)
    pred_labels = extract_tail_labels_topk(raw, k=3) or []
    return pred_labels, retrieved_facts, raw


def extract_tail_candidates(retrieved_facts: list[str], head_label: str, rel_label: str):
    """
    Returns unique tail labels (strings) extracted from facts in format:
      In YEAR, HEAD REL TAIL.
    """
    candidates = []
    key = f"{head_label} {rel_label} "

    for f in retrieved_facts:
        if key not in f:
            continue

        tail = f.split(key, 1)[1].strip()

        # delete dot at the end
        tail = re.sub(r"\.\s*$", "", tail)

        tail = tail.strip()

        if tail:
            candidates.append(tail)

    # unique candidates with retained order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

def rag_answer_open_world_top3(head_id: str, rel_id: str, ts: int, k=24):
    head_label = qid_to_label.get(head_id, head_id)
    rel_label  = pid_to_label.get(rel_id, rel_id)

    retrieved_facts = retrieve_facts_new(head_id, rel_id, ts, k=k)

    user_prompt = (
        f"Query:\nTime: {ts}\nHead: {head_label}\nRelation: {rel_label}\n\n"
        "Retrieved facts:\n" + "\n".join([f"- {f}" for f in retrieved_facts]) + "\n\n"
        "Task:\n"
        "Predict up to 3 most likely tail entity labels (best→worst).\n"
        "The correct tail may NOT be in the retrieved facts.\n"
        'Return ONLY JSON: {"tail_labels": ["...","...","..."]}\n'
        'If none, return {"tail_labels": []}.'
    )

    raw = qwen_llm_open_top3(user_prompt)
    pred_labels = extract_tail_labels_topk(raw, k=3)  # None / [] / ["a","b","c"]
    return pred_labels, retrieved_facts, raw


def rag_answer_hybrid_top3(head_id: str, rel_id: str, ts: int, k=24):
    # try candidate-only
    head_label = qid_to_label.get(head_id, head_id)
    rel_label  = pid_to_label.get(rel_id, rel_id)

    retrieved_facts = retrieve_facts_new(head_id, rel_id, ts, k=k)

    candidates = extract_tail_candidates(retrieved_facts, head_label, rel_label)

    if candidates:
        # candidate-only top3 
        pred_labels, retrieved_facts2, raw = rag_answer_candidates_top3(head_id, rel_id, ts, k=k)
        return pred_labels, retrieved_facts2, raw #, "candidates"

    # fallback: open-world top3
    pred_labels, retrieved_facts2, raw = rag_answer_open_world_top3(head_id, rel_id, ts, k=k)
    return pred_labels, retrieved_facts2, raw #, "open_world"


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_tail_labels_topk(raw: str, k=3):
    if not raw:
        return None

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    m = _JSON_RE.search(cleaned)
    if not m:
        return None

    block = m.group(0)
    try:
        obj = json.loads(block)
    except json.JSONDecodeError: 
        # just in case for apostrophes
        try:
            obj = json.loads(block.replace("'", '"'))
        except json.JSONDecodeError:
            return None

    labels = obj.get("tail_labels", None)
    if labels is None:
        return None
    if not isinstance(labels, list):
        return None

    labels = [x.strip() for x in labels if isinstance(x, str) and x.strip()]
    return labels[:k]


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


shuffled_df = test_df.sample(frac=1.0, random_state=42).reset_index(drop=False)
shuffled_df = shuffled_df.rename(columns={"index": "orig_idx"})

def eval_checkpoint_generic(
    df,
    rag_answer,
    out_jsonl="runs/eval_results.jsonl",
    k_retrieval=24,
    max_examples=None,
    start_from="auto",
    log_every=100,
    debug_first_n=3,
    store_raw=False,  
):
    Path(os.path.dirname(out_jsonl) or ".").mkdir(parents=True, exist_ok=True)
    N = len(df) if max_examples is None else min(len(df), max_examples)

    start_i = count_lines(out_jsonl) if start_from == "auto" else int(start_from)
    print(f"Will process [{start_i} .. {N-1}] (N={N}). Output -> {out_jsonl}")

    debug_printed = 0
    t0 = time.time()

    for i in range(start_i, N):
        row = df.iloc[i]
        ts = int(row["ts"])
        head_id = str(row["head"])
        rel_id  = str(row["relation_type"])
        gold_tail = str(row["tail"])

        orig_idx = int(row["orig_idx"])
        
        # --- model call ---
        pred_labels, retrieved_facts, raw = rag_answer(head_id, rel_id, ts, k=k_retrieval)
        # If rag_answer returns raw in format tail_label (single), not tail_labels,
        # we parse it here:
        if pred_labels is None:
            pred_labels = extract_tail_labels_topk(raw, k=3)

        status = "ok"
        if pred_labels is None:
            status = "parse_fail"
        elif len(pred_labels) == 0:
            status = "empty"

        if status == "parse_fail" and debug_printed < debug_first_n:
            print("RAW (parse_fail example):\n", raw)
            debug_printed += 1

        rec = {
            "i": i,
            "orig_idx": orig_idx,
            "ts": ts,
            "head": head_id,
            "rel": rel_id,
            "gold_tail": gold_tail,
            "pred_labels": pred_labels,   # None / [] / ["a","b","c"]
            "status": status,
            "k": k_retrieval,
        }
        if store_raw:
            rec["raw"] = raw

        append_jsonl(out_jsonl, rec)

        if log_every and (i + 1) % log_every == 0:
            dt = time.time() - t0
            speed = (i + 1 - start_i) / max(dt, 1e-9)
            eta = (N - (i + 1)) / max(speed, 1e-9)
            print(f"{i+1}/{N} saved | speed={speed:.3f} ex/s | ETA~{eta/60:.1f} min")

    print("Done.")


eval_checkpoint_generic(
    shuffled_df,                        # full shuffled test
    rag_answer=rag_answer_hybrid_top3,
    out_jsonl="runs/hybrid_k24.jsonl",
    k_retrieval=24,
    start_from="auto",
    log_every=100,
    store_raw=False
)