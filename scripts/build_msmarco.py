import os
import sys
import json
import time
import tarfile
import shutil
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from scripts.build_datasets import ColBERTv2Encoder, save_dataset


def main():
    raw_dir = os.path.expanduser("~/maxbound/raw_msmarco")
    out_dir = os.path.expanduser("~/maxbound/data_mv/msmarco")
    model_dir = os.path.expanduser("~/maxbound/colbertv2")

    tar_path = os.path.join(raw_dir, "collectionandqueries.tar.gz")
    dev_qrels_beir = os.path.join(raw_dir, "beir_dev.tsv")

    print("Extracting collectionandqueries.tar.gz...")
    with tarfile.open(tar_path, "r:gz") as tf:
        names = tf.getnames()
        print("Archive members:", names)
        tf.extractall(raw_dir)

    # Locate collection.tsv, queries, and qrels
    coll_path = None
    q_path = None
    qrel_path = None
    for root, _, files in os.walk(raw_dir):
        for fn in files:
            fp = os.path.join(root, fn)
            if fn == "collection.tsv":
                coll_path = fp
            elif fn in ("queries.dev.small.tsv", "queries.dev.tsv"):
                if q_path is None or "small" in fn:
                    q_path = fp
            elif fn in ("qrels.dev.small.tsv", "qrels.dev.tsv"):
                if qrel_path is None or "small" in fn:
                    qrel_path = fp

    if qrel_path is None and os.path.exists(dev_qrels_beir):
        qrel_path = dev_qrels_beir
    assert coll_path is not None and q_path is not None and qrel_path is not None, \
        f"Missing files: coll={coll_path}, q={q_path}, qrel={qrel_path}"

    print(f"Using collection={coll_path}, queries={q_path}, qrels={qrel_path}")

    # Load queries
    qid_to_txt = {}
    with open(q_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                qid_to_txt[parts[0].strip()] = parts[1].strip()

    # Load qrels (handle both 4-col TREC format: qid 0 pid rel and 3-col BEIR format: qid pid rel)
    qrels_raw = {}
    all_pos_pids = set()
    with open(qrel_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] == "query-id":
                continue
            if len(parts) >= 4:
                qid, pid, rel = parts[0].strip(), parts[2].strip(), float(parts[3])
            else:
                qid, pid, rel = parts[0].strip(), parts[1].strip(), float(parts[2])
            if rel > 0 and qid in qid_to_txt:
                qrels_raw.setdefault(qid, {})[pid] = rel
                all_pos_pids.add(pid)

    print(f"Loaded {len(qid_to_txt)} dev queries, {len(qrels_raw)} queries with qrels, {len(all_pos_pids)} unique positive passages.")

    # Deterministic selection of 512 official dev queries with positive qrels
    rng = np.random.default_rng(2027)
    eval_qids_all = sorted(qrels_raw.keys())
    num_queries = min(512, len(eval_qids_all))
    chosen_q_idx = rng.choice(len(eval_qids_all), size=num_queries, replace=False)
    chosen_qids = [eval_qids_all[i] for i in sorted(chosen_q_idx)]
    queries_txt = [qid_to_txt[qid] for qid in chosen_qids]

    # Retain 100% of positive passages across ALL dev qrels (~7.4K passages) + deterministic sample up to 500,000 passages
    target_docs = 500000
    need_other = target_docs - len(all_pos_pids)
    print("Scanning 8.84M official MS MARCO passages to select 500,000 passages (incl. 100% dev positives)...")
    # Pre-sample line indices in C/NumPy (~0.15s) instead of per-line RNG calls!
    sampled_line_set = set(int(x) for x in rng.choice(8841823, size=min(8841823, need_other + 30000), replace=False))
    pos_docs = []
    other_docs = []
    total_scanned = 0

    with open(coll_path, "r", encoding="utf-8", errors="replace") as f:
        for line_idx, line in enumerate(f):
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) < 2:
                continue
            pid, txt = parts[0].strip(), parts[1].strip()
            if not txt:
                continue
            total_scanned += 1
            if pid in all_pos_pids:
                pos_docs.append((pid, txt))
            elif line_idx in sampled_line_set and len(other_docs) < need_other:
                other_docs.append((pid, txt))

    all_selected = pos_docs + other_docs[:max(0, target_docs - len(pos_docs))]
    # Shuffle deterministically so positive passages are uniformly distributed across corpus IDs (zero positional bias!)
    perm = rng.permutation(len(all_selected))
    docs_txt = []
    pid_to_new_idx = {}
    for new_i, old_i in enumerate(perm):
        pid, txt = all_selected[int(old_i)]
        pid_to_new_idx[pid] = new_i
        docs_txt.append(txt)

    qrels = {}
    for qi, qid in enumerate(chosen_qids):
        rel_map = {}
        for pid, rel in qrels_raw[qid].items():
            if pid in pid_to_new_idx:
                rel_map[str(pid_to_new_idx[pid])] = rel
        if rel_map:
            qrels[str(qi)] = rel_map

    print(f"Selected {len(docs_txt)} passages (scanned {total_scanned}), {len(queries_txt)} official dev queries ({len(qrels)} mapped qrels). Encoding with ColBERTv2.0 on A100...")
    t0 = time.time()
    enc = ColBERTv2Encoder(model_dir, device="cuda:1", query_maxlen=32, doc_maxlen=300)
    d_vecs, d_lens = enc.encode_docs(docs_txt, maxlen=300, batch_size=512)
    q_vecs, q_lens = enc.encode_queries(queries_txt, maxlen=32, batch_size=256)
    enc_s = round(time.time() - t0, 2)

    save_dataset(
        out_dir, d_vecs, d_lens, q_vecs, q_lens, qrels,
        {
            "source": "Microsoft MS MARCO Passage Ranking v1 (collectionandqueries.tar.gz, 500K passages incl. 100% dev positives + 512 official dev queries)",
            "total_collection_scanned": total_scanned,
            "encoder": "colbertv2.0",
            "encode_s": enc_s,
        }
    )

    # Also build msmarco_bgem3 (50K subset incl. 100% positives of the 512 queries) OR clean up raw archive immediately!
    print("Cleaning up temporary raw_msmarco archive to keep disk usage compact...")
    shutil.rmtree(raw_dir, ignore_errors=True)
    print("MS MARCO Passage dataset build COMPLETE!")


if __name__ == "__main__":
    main()
