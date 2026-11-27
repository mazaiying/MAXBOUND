import os
import sys
import json
import string
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel

sys.path.insert(0, os.path.expanduser("~/maxbound"))


class ColBERTv2Encoder:
    def __init__(self, model_dir, device="cuda:1", query_maxlen=32, doc_maxlen=180):
        self.device = device
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.bert = BertModel.from_pretrained(model_dir, local_files_only=True)
        sd = torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location="cpu", weights_only=False)
        # Inspect linear key
        lin_w_key = [k for k in sd.keys() if "linear.weight" in k][0]
        w = sd[lin_w_key]  # [128, 768]
        self.linear = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        self.linear.weight.data.copy_(w)
        if lin_w_key.replace("weight", "bias") in sd:
            b = sd[lin_w_key.replace("weight", "bias")]
            self.linear = torch.nn.Linear(w.shape[1], w.shape[0], bias=True)
            self.linear.weight.data.copy_(w)
            self.linear.bias.data.copy_(b)
        self.bert.to(device).half().eval()
        self.linear.to(device).half().eval()

        # Punctuation token IDs to mask out for documents (standard ColBERTv2)
        skiplist_ids = set()
        for p in string.punctuation:
            tid = self.tok.encode(p, add_special_tokens=False)
            if len(tid) == 1:
                skiplist_ids.add(tid[0])
        for tid in [self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id]:
            if tid is not None:
                skiplist_ids.add(tid)
        self.skiplist = torch.zeros(self.tok.vocab_size, dtype=torch.bool, device=device)
        for tid in skiplist_ids:
            if 0 <= tid < self.tok.vocab_size:
                self.skiplist[tid] = True

        # Special marker IDs: ColBERTv2 uses [unused0] (id=1) for [Q] and [unused1] (id=2) for [D]
        self.Q_marker_id = self.tok.convert_tokens_to_ids("[unused0]")
        self.D_marker_id = self.tok.convert_tokens_to_ids("[unused1]")

    @torch.no_grad()
    def encode_queries(self, texts, maxlen=None, batch_size=256):
        maxlen = maxlen or self.query_maxlen
        all_vecs = []
        all_lens = []
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            enc = self.tok(
                batch,
                padding="max_length",
                truncation=True,
                max_length=maxlen,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            # Insert [Q] marker right after [CLS]
            ids[:, 1] = self.Q_marker_id
            # Standard ColBERTv2 query augmentation: replace [PAD] with [MASK]
            ids[ids == self.tok.pad_token_id] = self.tok.mask_token_id
            mask[:] = 1
            out = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
            proj = F.normalize(self.linear(out), p=2, dim=-1)
            all_vecs.append(proj.half().cpu().numpy())
            all_lens.extend([maxlen] * len(batch))
        return np.concatenate(all_vecs, axis=0).reshape(-1, proj.shape[-1]), np.array(all_lens, dtype=np.int64)

    @torch.no_grad()
    def encode_docs(self, texts, maxlen=None, batch_size=256):
        maxlen = maxlen or self.doc_maxlen
        vecs_list = []
        lens_list = []
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=maxlen,
                return_tensors="pt",
            )
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            if ids.shape[1] > 1:
                ids[:, 1] = self.D_marker_id
            out = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
            proj = F.normalize(self.linear(out), p=2, dim=-1).half()
            # Keep non-pad and non-skiplist tokens (keep at least 1 token if doc is empty)
            keep_mask = (mask > 0) & (~self.skiplist[ids])
            proj_cpu = proj.cpu().numpy()
            keep_cpu = keep_mask.cpu().numpy()
            for i in range(len(batch)):
                sel = proj_cpu[i][keep_cpu[i]]
                if len(sel) == 0:
                    sel = proj_cpu[i][:1]
                vecs_list.append(sel)
                lens_list.append(len(sel))
        return np.concatenate(vecs_list, axis=0), np.array(lens_list, dtype=np.int64)


def save_dataset(out_dir, d_vecs, d_lens, q_vecs, q_lens, qrels, extra_meta=None):
    os.makedirs(out_dir, exist_ok=True)
    d_offsets = np.zeros(len(d_lens) + 1, dtype=np.int64)
    np.cumsum(d_lens, out=d_offsets[1:])
    q_offsets = np.zeros(len(q_lens) + 1, dtype=np.int64)
    np.cumsum(q_lens, out=q_offsets[1:])
    np.save(os.path.join(out_dir, "doc_tokens.fp16.npy"), d_vecs.astype(np.float16))
    np.save(os.path.join(out_dir, "doc_offsets.int64.npy"), d_offsets)
    np.save(os.path.join(out_dir, "query_tokens.fp16.npy"), q_vecs.astype(np.float16))
    np.save(os.path.join(out_dir, "query_offsets.int64.npy"), q_offsets)
    with open(os.path.join(out_dir, "qrels.json"), "w") as f:
        json.dump(qrels, f)
    meta = {
        "num_docs": int(len(d_lens)),
        "total_doc_tokens": int(d_offsets[-1]),
        "num_queries": int(len(q_lens)),
        "total_query_tokens": int(q_offsets[-1]),
        "dim": int(d_vecs.shape[1]),
        "doc_len_mean": float(np.mean(d_lens)),
        "doc_len_std": float(np.std(d_lens)),
        "doc_len_cv": float(np.std(d_lens) / max(np.mean(d_lens), 1e-9)),
        "doc_len_min": int(np.min(d_lens)),
        "doc_len_p50": float(np.percentile(d_lens, 50)),
        "doc_len_p95": float(np.percentile(d_lens, 95)),
        "doc_len_max": int(np.max(d_lens)),
        **(extra_meta or {}),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {out_dir}: {meta['num_docs']} docs, {meta['total_doc_tokens']} toks, CV={meta['doc_len_cv']:.2f}, {meta['num_queries']} queries")


def load_beir_dir(beir_dir):
    docs_id, docs_txt = [], []
    id_to_idx = {}
    with open(os.path.join(beir_dir, "corpus.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            did = str(obj["_id"])
            txt = (obj.get("title", "") + " " + obj.get("text", "")).strip()
            id_to_idx[did] = len(docs_id)
            docs_id.append(did)
            docs_txt.append(txt)
    q_id_to_txt = {}
    with open(os.path.join(beir_dir, "queries.jsonl")) as f:
        for line in f:
            obj = json.loads(line)
            q_id_to_txt[str(obj["_id"])] = obj["text"].strip()
    qrels_raw = {}
    with open(os.path.join(beir_dir, "test.tsv")) as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], float(parts[2])
            if did in id_to_idx and score > 0:
                qrels_raw.setdefault(qid, {})[str(id_to_idx[did])] = score
    # Keep queries that have qrels (or up to 1000 total queries)
    eval_qids = [qid for qid in qrels_raw.keys() if qid in q_id_to_txt]
    queries_txt = [q_id_to_txt[qid] for qid in eval_qids]
    qrels = {str(i): qrels_raw[qid] for i, qid in enumerate(eval_qids)}
    # If fewer than 256 queries (e.g., trec-covid has 50 official), append extra queries for QPS benchmarking while keeping qrels for official ones
    if len(queries_txt) < 256:
        used = set(eval_qids)
        for qid, qt in q_id_to_txt.items():
            if qid not in used:
                queries_txt.append(qt)
                if len(queries_txt) >= 512:
                    break
    return docs_txt, queries_txt, qrels


def load_hotpotqa(path, max_examples=50000, num_queries=1000):
    with open(path) as f:
        data = json.load(f)
    title_to_text = {}
    doc_idx_map = {}
    docs_txt = []
    queries_txt = []
    qrels = {}
    for ex in data[:max_examples]:
        for title, sents in ex["context"]:
            if title not in doc_idx_map:
                doc_idx_map[title] = len(docs_txt)
                docs_txt.append(title + ". " + " ".join(sents))
        if len(queries_txt) < num_queries:
            qi = len(queries_txt)
            queries_txt.append(ex["question"])
            rel_titles = set(t for t, _ in ex.get("supporting_facts", []))
            rel = {}
            for t in rel_titles:
                if t in doc_idx_map:
                    rel[str(doc_idx_map[t])] = 1.0
            if rel:
                qrels[str(qi)] = rel
    return docs_txt, queries_txt, qrels


def load_longbench_docs(lb_dir, max_docs=25000, num_queries=512):
    docs_txt = []
    queries_txt = []
    qrels = {}
    for fn in ["hotpotqa_e.jsonl", "2wikimqa_e.jsonl", "musique.jsonl", "triviaqa_e.jsonl"]:
        p = os.path.join(lb_dir, fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                obj = json.loads(line)
                ctx = obj.get("context", "")
                # Split into variable-length documents (100 to 900 words) to capture extreme length skew!
                words = ctx.split()
                pos = 0
                doc_ids_for_ex = []
                step_choices = [120, 220, 380, 520, 700]
                while pos < len(words) and len(docs_txt) < max_docs:
                    wlen = step_choices[(len(docs_txt) + pos) % len(step_choices)]
                    chunk = " ".join(words[pos:pos + wlen])
                    if len(chunk.strip()) > 30:
                        doc_ids_for_ex.append(len(docs_txt))
                        docs_txt.append(chunk)
                    pos += wlen
                if len(queries_txt) < num_queries and obj.get("input") and doc_ids_for_ex:
                    qi = len(queries_txt)
                    queries_txt.append(obj["input"])
                    qrels[str(qi)] = {str(did): 1.0 for did in doc_ids_for_ex[:3]}
                if len(docs_txt) >= max_docs:
                    break
    return docs_txt, queries_txt, qrels


def load_scale50m(parquet_path, probe_path, max_docs=500000, num_queries=512):
    df = pd.read_parquet(parquet_path, columns=["text"])
    texts = df["text"].dropna().astype(str).tolist()[:max_docs]
    df_q = pd.read_parquet(probe_path)
    q_col = "question" if "question" in df_q.columns else ("query" if "query" in df_q.columns else df_q.columns[0])
    queries_txt = df_q[q_col].dropna().astype(str).tolist()[:num_queries]
    return texts, queries_txt, {}


def encode_bgem3(bge_dir, docs_txt, queries_txt, qrels, out_dir, device="cuda:1"):
    from transformers import AutoModel
    tok = AutoTokenizer.from_pretrained(bge_dir, local_files_only=True)
    model = AutoModel.from_pretrained(bge_dir, local_files_only=True).to(device).half().eval()

    @torch.no_grad()
    def _enc(texts, maxlen, bs=64):
        vecs, lens = [], []
        for s in range(0, len(texts), bs):
            b = texts[s:s + bs]
            enc = tok(b, padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            out = model(input_ids=ids, attention_mask=mask).last_hidden_state
            proj = F.normalize(out, p=2, dim=-1).half()
            p_cpu = proj.cpu().numpy()
            m_cpu = mask.cpu().numpy()
            for i in range(len(b)):
                sel = p_cpu[i][m_cpu[i] > 0]
                if len(sel) == 0:
                    sel = p_cpu[i][:1]
                vecs.append(sel)
                lens.append(len(sel))
        return np.concatenate(vecs, axis=0), np.array(lens, dtype=np.int64)

    d_vecs, d_lens = _enc(docs_txt, 180, bs=64)
    q_vecs, q_lens = _enc(queries_txt, 32, bs=128)
    save_dataset(out_dir, d_vecs, d_lens, q_vecs, q_lens, qrels, {"encoder": "BAAI/bge-m3", "dim": 1024})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    model_dir = os.path.expanduser("~/maxbound/colbertv2")
    data_root = os.path.expanduser("~/maxbound/data_mv")
    beir_root = os.path.expanduser("~/maxbound/raw_beir")
    enc = ColBERTv2Encoder(model_dir, device="cuda:1", query_maxlen=32, doc_maxlen=350)

    def want(name):
        return args.only is None or name in args.only

    # 1-6: BEIR datasets
    for ds in ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid"]:
        ds_name = ds.replace("-", "_")
        if not want(ds_name):
            continue
        bdir = os.path.join(beir_root, ds)
        if not os.path.exists(os.path.join(bdir, "corpus.jsonl")):
            print(f"Skipping {ds}, not found at {bdir}")
            continue
        t0 = time.time()
        docs_txt, queries_txt, qrels = load_beir_dir(bdir)
        q_max = 64 if ds == "arguana" else 32
        d_vecs, d_lens = enc.encode_docs(docs_txt, maxlen=350, batch_size=256)
        q_vecs, q_lens = enc.encode_queries(queries_txt, maxlen=q_max, batch_size=256)
        save_dataset(os.path.join(data_root, ds_name), d_vecs, d_lens, q_vecs, q_lens, qrels,
                     {"source": f"mteb/{ds}", "encoder": "colbertv2.0", "encode_s": round(time.time() - t0, 2)})

    # 7: HotpotQA (~250K passages, real supporting facts)
    if want("hotpotqa"):
        t0 = time.time()
        docs_txt, queries_txt, qrels = load_hotpotqa(
            os.path.expanduser("~/cecomp_20260908/data/raw/hotpot_train_v1.1.json"),
            max_examples=40000, num_queries=1000
        )
        d_vecs, d_lens = enc.encode_docs(docs_txt, maxlen=350, batch_size=256)
        q_vecs, q_lens = enc.encode_queries(queries_txt, maxlen=32, batch_size=256)
        save_dataset(os.path.join(data_root, "hotpotqa"), d_vecs, d_lens, q_vecs, q_lens, qrels,
                     {"source": "hotpot_train_v1.1", "encoder": "colbertv2.0", "encode_s": round(time.time() - t0, 2)})

    # 8: LongBench-Ext (extreme length skew up to 512 tokens)
    if want("longbench"):
        t0 = time.time()
        docs_txt, queries_txt, qrels = load_longbench_docs(
            os.path.expanduser("~/kv_decision_qwen3b_exp_v1/longbench_data_v1/data"),
            max_docs=25000, num_queries=512
        )
        enc_long = ColBERTv2Encoder(model_dir, device="cuda:1", query_maxlen=32, doc_maxlen=512)
        d_vecs, d_lens = enc_long.encode_docs(docs_txt, maxlen=512, batch_size=96)
        q_vecs, q_lens = enc_long.encode_queries(queries_txt, maxlen=32, batch_size=256)
        save_dataset(os.path.join(data_root, "longbench"), d_vecs, d_lens, q_vecs, q_lens, qrels,
                     {"source": "LongBench", "encoder": "colbertv2.0", "encode_s": round(time.time() - t0, 2)})

    # 9: Scale-50M (~980K real WildChat passages, ~45M tokens, ~11 GB FP16)
    if want("scale_50m"):
        t0 = time.time()
        docs_txt, queries_txt, qrels = load_scale50m(
            os.path.expanduser("~/agentmembench-kdd27/data/scale1m/distractors_1m.parquet"),
            os.path.expanduser("~/agentmembench-kdd27/data/scale1m/probe.parquet"),
            max_docs=980000, num_queries=512
        )
        d_vecs, d_lens = enc.encode_docs(docs_txt, maxlen=350, batch_size=512)
        q_vecs, q_lens = enc.encode_queries(queries_txt, maxlen=32, batch_size=256)
        save_dataset(os.path.join(data_root, "scale_50m"), d_vecs, d_lens, q_vecs, q_lens, qrels,
                     {"source": "WildChat-Scale1M", "encoder": "colbertv2.0", "encode_s": round(time.time() - t0, 2)})

    # 10: SciFact-BGEM3 (d=1024)
    if want("scifact_bgem3"):
        bdir = os.path.join(beir_root, "scifact")
        docs_txt, queries_txt, qrels = load_beir_dir(bdir)
        bge_dir = os.environ.get("BGE_M3_DIR", "BAAI/bge-m3")
        encode_bgem3(bge_dir, docs_txt, queries_txt, qrels, os.path.join(data_root, "scifact_bgem3"), device="cuda:1")
