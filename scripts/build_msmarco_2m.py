import os
import sys
import json
import time
import shutil
import subprocess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from scripts.build_datasets import ColBERTv2Encoder


def run_gpu_worker(out_dir: str, texts_mmap_pkl: str, doc_start: int, doc_end: int, device_id: int, batch_size: int):
    import pickle
    torch.cuda.set_device(device_id)
    dev_str = f"cuda:{device_id}"
    model_dir = os.path.expanduser("~/maxbound/colbertv2")
    enc = ColBERTv2Encoder(model_dir, device=dev_str, query_maxlen=32, doc_maxlen=300)

    with open(texts_mmap_pkl, "rb") as f:
        docs_sub = pickle.load(f)
    assert len(docs_sub) == (doc_end - doc_start)

    d_offsets = np.load(os.path.join(out_dir, "doc_offsets.int64.npy"))
    mmap_vecs = np.lib.format.open_memmap(os.path.join(out_dir, "doc_tokens.fp16.npy"), mode="r+")

    # Process in mega-chunks of 16,384 docs; sort within mega-chunk by char length for minimal padding waste!
    n_local = len(docs_sub)
    t0 = time.time()
    for mc_s in range(0, n_local, 16384):
        mc_e = min(mc_s + 16384, n_local)
        sub_txts = docs_sub[mc_s:mc_e]
        order = np.argsort([len(t) for t in sub_txts], kind="stable")

        for b_s in range(0, len(order), batch_size):
            b_idx = order[b_s:b_s + batch_size]
            batch_txts = [sub_txts[int(k)] for k in b_idx]
            tk = enc.tok(batch_txts, padding=True, truncation=True, max_length=300, return_tensors="pt")
            ids = tk["input_ids"].to(dev_str, non_blocking=True)
            mask = tk["attention_mask"].to(dev_str, non_blocking=True)
            if ids.shape[1] > 1:
                ids[:, 1] = enc.D_marker_id
            with torch.no_grad():
                out = enc.bert(input_ids=ids, attention_mask=mask).last_hidden_state
                proj = F.normalize(enc.linear(out), p=2, dim=-1).half()
                keep_mask = (mask > 0) & (~enc.skiplist[ids])
            proj_cpu = proj.cpu().numpy()
            keep_cpu = keep_mask.cpu().numpy()

            for loc_i, orig_in_mc in enumerate(b_idx):
                g_doc = doc_start + mc_s + int(orig_in_mc)
                sel = proj_cpu[loc_i][keep_cpu[loc_i]]
                if len(sel) == 0:
                    sel = proj_cpu[loc_i][:1]
                s_off = int(d_offsets[g_doc])
                e_off = int(d_offsets[g_doc + 1])
                assert (e_off - s_off) == len(sel), f"Len mismatch doc {g_doc}: expected {e_off-s_off}, got {len(sel)}"
                mmap_vecs[s_off:e_off] = sel

        done = mc_e
        el = time.time() - t0
        print(f"[GPU {device_id}] {done:,}/{n_local:,} docs encoded ({done / max(el, 1e-3):.0f} docs/s)", flush=True)

    mmap_vecs.flush()


def main():
    import pickle
    raw_dir = os.path.expanduser("~/maxbound/raw_msmarco")
    out_dir = os.path.expanduser("~/maxbound/data_mv/msmarco_2m")
    model_dir = os.path.expanduser("~/maxbound/colbertv2")

    if os.path.exists(os.path.join(out_dir, "meta.json")) and os.path.exists(os.path.join(out_dir, "doc_tokens.fp16.npy")):
        print("msmarco_2m already exists, verifying...")
        with open(os.path.join(out_dir, "meta.json")) as f:
            m = json.load(f)
        if m.get("num_docs") == 2000000:
            print("Verified 2,000,000 docs ready:", m)
            return

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
            elif fn in ("qrels.dev.small.tsv", "qrels.dev.tsv", "beir_dev.tsv"):
                if qrel_path is None or "small" in fn:
                    qrel_path = fp

    if coll_path is None:
        import tarfile
        tar_path = os.path.join(raw_dir, "collectionandqueries.tar.gz")
        assert os.path.exists(tar_path), f"Missing {tar_path}"
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(raw_dir)
        for root, _, files in os.walk(raw_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                if fn == "collection.tsv":
                    coll_path = fp
                elif fn in ("queries.dev.small.tsv", "queries.dev.tsv"):
                    if q_path is None or "small" in fn:
                        q_path = fp
                elif fn in ("qrels.dev.small.tsv", "qrels.dev.tsv", "beir_dev.tsv"):
                    if qrel_path is None or "small" in fn:
                        qrel_path = fp

    assert coll_path and q_path and qrel_path, f"Missing raw files: {coll_path}, {q_path}, {qrel_path}"

    # Remove compressed tar archive immediately to save 1.0 GB before allocating msmarco_2m!
    tar_path = os.path.join(raw_dir, "collectionandqueries.tar.gz")
    if os.path.exists(tar_path):
        os.remove(tar_path)

    qid_to_txt = {}
    with open(q_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                qid_to_txt[parts[0].strip()] = parts[1].strip()

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

    rng = np.random.default_rng(2028)
    eval_qids_all = sorted(qrels_raw.keys())
    num_queries = min(512, len(eval_qids_all))
    chosen_q_idx = rng.choice(len(eval_qids_all), size=num_queries, replace=False)
    chosen_qids = [eval_qids_all[i] for i in sorted(chosen_q_idx)]
    queries_txt = [qid_to_txt[qid] for qid in chosen_qids]

    target_docs = 2_000_000
    need_other = target_docs - len(all_pos_pids)
    print(f"Scanning 8.84M official MS MARCO passages to select {target_docs:,} passages (incl. 100% dev positives)...")
    sampled_mask = np.zeros(8900000, dtype=bool)
    sampled_idx = rng.choice(8841823, size=min(8841823, need_other + 60000), replace=False)
    sampled_mask[sampled_idx] = True

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
            elif line_idx < len(sampled_mask) and sampled_mask[line_idx] and len(other_docs) < need_other:
                other_docs.append((pid, txt))

    all_selected = pos_docs + other_docs[:max(0, target_docs - len(pos_docs))]
    assert len(all_selected) == target_docs, f"Expected {target_docs}, got {len(all_selected)}"

    perm = rng.permutation(target_docs)
    docs_txt = [None] * target_docs
    pid_to_new_idx = {}
    for new_i, old_i in enumerate(perm):
        pid, txt = all_selected[int(old_i)]
        pid_to_new_idx[pid] = new_i
        docs_txt[new_i] = txt
    del pos_docs, other_docs, all_selected

    # Free raw_msmarco (2.9 GB) BEFORE pre-allocating 31 GB doc_tokens.fp16.npy to stay strictly < 91 GB!
    shutil.rmtree(raw_dir, ignore_errors=True)

    qrels = {}
    for qi, qid in enumerate(chosen_qids):
        rel_map = {}
        for pid, rel in qrels_raw[qid].items():
            if pid in pid_to_new_idx:
                rel_map[str(pid_to_new_idx[pid])] = rel
        if rel_map:
            qrels[str(qi)] = rel_map

    os.makedirs(out_dir, exist_ok=True)
    t_start = time.time()

    # Step 1: Fast Rust-backed pre-tokenization length pass to allocate exact contiguous FP16 mmap (0 temp duplicates!)
    print("Pre-computing exact ColBERTv2 token lengths for 2,000,000 passages...")
    enc_q = ColBERTv2Encoder(model_dir, device="cuda:1", query_maxlen=32, doc_maxlen=300)
    skiplist_np = enc_q.skiplist.cpu().numpy()
    d_marker = enc_q.D_marker_id

    d_lens = np.empty(target_docs, dtype=np.int64)
    for s in range(0, target_docs, 16384):
        e = min(s + 16384, target_docs)
        tk = enc_q.tok(docs_txt[s:e], padding=True, truncation=True, max_length=300, return_tensors="np")
        ids = tk["input_ids"]
        mask = tk["attention_mask"]
        if ids.shape[1] > 1:
            ids[:, 1] = d_marker
        cnt = ((mask > 0) & (~skiplist_np[ids])).sum(axis=1)
        d_lens[s:e] = np.maximum(cnt, 1)

    d_offsets = np.zeros(target_docs + 1, dtype=np.int64)
    np.cumsum(d_lens, out=d_offsets[1:])
    total_toks = int(d_offsets[-1])
    print(f"Total tokens across 2,000,000 passages = {total_toks:,} ({total_toks * 256 / (1024**3):.2f} GiB FP16)")

    q_vecs, q_lens = enc_q.encode_queries(queries_txt, maxlen=32, batch_size=256)
    del enc_q
    torch.cuda.empty_cache()

    q_offsets = np.zeros(len(q_lens) + 1, dtype=np.int64)
    np.cumsum(q_lens, out=q_offsets[1:])
    np.save(os.path.join(out_dir, "doc_offsets.int64.npy"), d_offsets)
    np.save(os.path.join(out_dir, "query_tokens.fp16.npy"), q_vecs.astype(np.float16))
    np.save(os.path.join(out_dir, "query_offsets.int64.npy"), q_offsets)
    with open(os.path.join(out_dir, "qrels.json"), "w") as f:
        json.dump(qrels, f)

    # Pre-create memory-mapped doc_tokens.fp16.npy
    mmap_fp16 = np.lib.format.open_memmap(
        os.path.join(out_dir, "doc_tokens.fp16.npy"), mode="w+", dtype=np.float16, shape=(total_toks, 128)
    )
    mmap_fp16.flush()
    del mmap_fp16

    # Split 2,000,000 docs across GPU 1 (A100: 1,150,000 docs) and GPU 0 (RTX 4080: 850,000 docs) in parallel!
    split_n = 1_150_000
    pkl1 = os.path.join(out_dir, "_tmp_docs1.pkl")
    pkl0 = os.path.join(out_dir, "_tmp_docs0.pkl")
    with open(pkl1, "wb") as f:
        pickle.dump(docs_txt[:split_n], f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(pkl0, "wb") as f:
        pickle.dump(docs_txt[split_n:], f, protocol=pickle.HIGHEST_PROTOCOL)
    del docs_txt

    p1 = subprocess.Popen([sys.executable, __file__, "--worker", out_dir, pkl1, "0", str(split_n), "1", "1024"])
    p0 = subprocess.Popen([sys.executable, __file__, "--worker", out_dir, pkl0, str(split_n), str(target_docs), "0", "512"])
    rc1 = p1.wait()
    rc0 = p0.wait()
    for pf in (pkl1, pkl0):
        if os.path.exists(pf):
            os.remove(pf)
    assert rc1 == 0 and rc0 == 0, f"Worker failed: rc1={rc1}, rc0={rc0}"

    enc_s = round(time.time() - t_start, 2)
    meta = {
        "num_docs": int(target_docs),
        "total_doc_tokens": int(total_toks),
        "num_queries": int(len(q_lens)),
        "total_query_tokens": int(q_offsets[-1]),
        "dim": 128,
        "doc_len_mean": float(np.mean(d_lens)),
        "doc_len_std": float(np.std(d_lens)),
        "doc_len_cv": float(np.std(d_lens) / max(np.mean(d_lens), 1e-9)),
        "doc_len_min": int(np.min(d_lens)),
        "doc_len_p50": float(np.percentile(d_lens, 50)),
        "doc_len_p95": float(np.percentile(d_lens, 95)),
        "doc_len_max": int(np.max(d_lens)),
        "source": "Microsoft MS MARCO Passage Ranking v1 (2,000,000 official passages incl. 100% dev positives + 512 official dev queries)",
        "total_collection_scanned": total_scanned,
        "encoder": "colbertv2.0",
        "encode_s": enc_s,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"MS MARCO 2M ({total_toks:,} tokens) COMPLETE in {enc_s:.1f}s!")


if __name__ == "__main__":
    if len(sys.argv) >= 8 and sys.argv[1] == "--worker":
        run_gpu_worker(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
    else:
        main()
