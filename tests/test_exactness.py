import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from src.oracle import exact_maxsim_oracle
from src.maxbound_engine import MaxBoundIndex
from src.baselines.baselines import B8FusedFP16TCExact, B4OfficialFastPlaid, B9WarpGPU, B10EmvbGPU
from bench.common import audit_exact_topk


def make_rand_norm(n, d=128, seed=2027):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def run_case(name, d_toks, d_offs, q_toks, q_offs, topk=10, device=0):
    oracle_ids, oracle_scores = exact_maxsim_oracle(
        q_toks.float(), q_offs, d_toks, d_offs, topk=topk, doc_batch_size=512, device=device, use_fp64=True
    )
    modes = [("int8", False, True), ("int8", False, False), ("int6", False, True), ("int4", False, True), ("int8", True, True)]
    for mode, offload, sbl in modes:
        idx = MaxBoundIndex(d_toks, d_offs, device=device, mode=mode, host_offload_fp16=offload, sort_by_length=sbl)
        pred_ids, pred_scores, info = idx.search(q_toks, q_offs, topk=topk, c_warm=16, return_tested_mask=True)
        aud = audit_exact_topk(pred_ids, pred_scores, oracle_ids, oracle_scores, tol=2e-3, tested_mask=info["tested_mask_orig"])
        assert aud["agreement"] == 1.0 and aud["num_mismatch"] == 0 and aud["false_negative_prunes"] == 0, \
            f"{name} ({mode}, offload={offload}, sort={sbl}, mask=True) FAILED: {aud}"
        # Also test fast zero-host-sync fused_stage2_verify path (return_tested_mask=False)!
        p_fast_ids, p_fast_sc, _ = idx.search(q_toks, q_offs, topk=topk, c_warm=16, return_tested_mask=False)
        aud_fast = audit_exact_topk(p_fast_ids, p_fast_sc, oracle_ids, oracle_scores, tol=2e-3)
        assert aud_fast["agreement"] == 1.0 and aud_fast["num_mismatch"] == 0, \
            f"{name} ({mode}, offload={offload}, sort={sbl}, fast_stage2) FAILED: {aud_fast}"
    print(f"[PASS] {name}")


def main():
    dev = 0
    if len(sys.argv) > 1:
        dev = int(sys.argv[1])
    print(f"Running 20 adversarial FP64-audited exactness & baseline tests on GPU {dev}...")

    rng = np.random.default_rng(2027)
    lens = rng.integers(16, 256, size=500)
    d_offs = np.zeros(501, dtype=np.int64); np.cumsum(lens, out=d_offs[1:])
    d_toks = torch.from_numpy(make_rand_norm(int(d_offs[-1]), 128, 1)).half()
    q_offs = np.arange(0, 33 * 16, 32, dtype=np.int64)
    q_toks = torch.from_numpy(make_rand_norm(int(q_offs[-1]), 128, 2)).half()
    run_case("01_skewed_lengths", d_toks, d_offs, q_toks, q_offs, device=dev)

    d2 = d_toks.clone()
    d2[0:5] = q_toks[0:5]
    run_case("02_collinear_outliers", d2, d_offs, q_toks, q_offs, device=dev)

    o3 = np.arange(0, 201 * 32, 32, dtype=np.int64)
    base_doc = torch.from_numpy(make_rand_norm(32, 128, 3)).half()
    d3 = base_doc.repeat(200, 1)
    run_case("03_all_duplicate_ties", d3, o3, q_toks[:128], q_offs[:5], device=dev)

    o4 = np.arange(0, 301, dtype=np.int64)
    d4 = torch.from_numpy(make_rand_norm(300, 128, 4)).half()
    run_case("04_single_token_docs", d4, o4, q_toks[:128], q_offs[:5], device=dev)

    lens5 = rng.integers(1024, 2048, size=64)
    o5 = np.zeros(65, dtype=np.int64); np.cumsum(lens5, out=o5[1:])
    d5 = torch.from_numpy(make_rand_norm(int(o5[-1]), 128, 5)).half()
    run_case("05_extreme_long_docs_2048", d5, o5, q_toks[:128], q_offs[:5], device=dev)

    qlens6 = rng.integers(5, 80, size=12)
    qo6 = np.zeros(13, dtype=np.int64); np.cumsum(qlens6, out=qo6[1:])
    qt6 = torch.from_numpy(make_rand_norm(int(qo6[-1]), 128, 6)).half()
    run_case("06_variable_query_lengths_gt64", d_toks, d_offs, qt6, qo6, device=dev)

    qt7 = qt6.clone() * torch.linspace(0.05, 10.0, qt6.shape[0]).unsqueeze(1).half()
    run_case("07_unnormalized_queries", d_toks, d_offs, qt7, qo6, device=dev)

    idx8 = MaxBoundIndex(d_toks, d_offs, device=dev, mode="int8")
    o_ids, o_sc = exact_maxsim_oracle(q_toks.float(), q_offs, d_toks, d_offs, topk=10, device=dev, use_fp64=True)
    p1_ids, p1_sc, i8a = idx8.search(q_toks, q_offs, topk=10, external_tau=torch.full((16,), -100.0), return_tested_mask=True)
    p2_ids, p2_sc, i8b = idx8.search(q_toks, q_offs, topk=10, external_tau=torch.from_numpy(o_sc[:, -1]), return_tested_mask=True)
    assert audit_exact_topk(p1_ids, p1_sc, o_ids, o_sc, tested_mask=i8a["tested_mask_orig"])["false_negative_prunes"] == 0
    assert audit_exact_topk(p2_ids, p2_sc, o_ids, o_sc, tested_mask=i8b["tested_mask_orig"])["false_negative_prunes"] == 0
    print("[PASS] 08_cross_gpu_tau_race_invariance")

    rev_q = torch.cat([q_toks[q_offs[i]:q_offs[i + 1]] for i in reversed(range(16))], dim=0)
    p_rev, ps_rev, _ = idx8.search(rev_q, q_offs, topk=10)
    assert audit_exact_topk(p_rev[::-1], ps_rev[::-1], o_ids, o_sc)["agreement"] == 1.0
    print("[PASS] 09_batch_order_invariance")

    o10 = d_offs[:8]
    d10 = d_toks[:int(o10[-1])]
    run_case("10_small_corpus_lt_topk", d10, o10, q_toks[:64], q_offs[:3], topk=10, device=dev)
    run_case("11_single_query_B1", d_toks, d_offs, q_toks[:32], q_offs[:2], topk=10, device=dev)

    cands = np.stack([rng.choice(500, size=100, replace=False).astype(np.int32) for _ in range(16)])
    or_ids, or_sc = exact_maxsim_oracle(q_toks.float(), q_offs, d_toks, d_offs, topk=10, device=dev, candidate_ids=cands, use_fp64=True)
    rr_ids, rr_sc, _ = idx8.search_rerank(q_toks, q_offs, cands, topk=10, c_warm=16)
    aud12 = audit_exact_topk(rr_ids, rr_sc, or_ids, or_sc)
    assert aud12["agreement"] == 1.0 and aud12["num_mismatch"] == 0, f"12_rerank FAILED: {aud12}"
    print("[PASS] 12_modeB_candidate_rerank")

    # 13. Native PTX D=1024 Operator P2-Fused (WQ=8/4 SMEM pass16 + 2-Slice Early Pruning) + Mode B rerank + host offload
    lens_m3 = rng.integers(20, 180, size=256)
    offs_m3 = np.zeros(257, dtype=np.int64); np.cumsum(lens_m3, out=offs_m3[1:])
    dt_m3 = torch.from_numpy(make_rand_norm(int(offs_m3[-1]), 1024, 13)).half()
    qo_m3 = np.arange(0, 17 * 32, 32, dtype=np.int64)
    qt_m3 = torch.from_numpy(make_rand_norm(int(qo_m3[-1]), 1024, 14)).half()
    or_m3_i, or_m3_s = exact_maxsim_oracle(qt_m3.float(), qo_m3, dt_m3, offs_m3, topk=10, device=dev, use_fp64=True)
    for offl in [False, True]:
        for fw1 in [False, True]:
            idx_m3 = MaxBoundIndex(dt_m3, offs_m3, device=dev, mode="int8", host_offload_fp16=offl)
            pm_i, pm_s, inf_m3 = idx_m3.search(qt_m3, qo_m3, topk=10, c_warm=16, force_w1=fw1, return_tested_mask=True)
            aud_m3 = audit_exact_topk(pm_i, pm_s, or_m3_i, or_m3_s, tol=5e-3, tested_mask=inf_m3["tested_mask_orig"])
            assert aud_m3["agreement"] == 1.0 and aud_m3["num_mismatch"] == 0 and aud_m3["false_negative_prunes"] == 0, \
                f"13_d1024 offl={offl} fw1={fw1} FAILED: {aud_m3}"
    # Also test D=1024 at B=1 and B=3
    idx_m3_gpu = MaxBoundIndex(dt_m3, offs_m3, device=dev, mode="int8", host_offload_fp16=False)
    for b_sub in [1, 3, 8, 16]:
        pi_s, ps_s, inf_s = idx_m3_gpu.search(qt_m3[:b_sub * 32], qo_m3[:b_sub + 1], topk=10, c_warm=16, return_tested_mask=True)
        aud_s = audit_exact_topk(pi_s, ps_s, or_m3_i[:b_sub], or_m3_s[:b_sub], tol=5e-3, tested_mask=inf_s["tested_mask_orig"])
        assert aud_s["agreement"] == 1.0 and aud_s["false_negative_prunes"] == 0, f"13_d1024 B={b_sub} FAILED: {aud_s}"
    cands_m3 = np.stack([rng.choice(256, size=80, replace=False).astype(np.int32) for _ in range(16)])
    orr_i, orr_s = exact_maxsim_oracle(qt_m3.float(), qo_m3, dt_m3, offs_m3, topk=10, device=dev, candidate_ids=cands_m3, use_fp64=True)
    prr_i, prr_s, _ = idx_m3_gpu.search_rerank(qt_m3, qo_m3, cands_m3, topk=10, c_warm=16)
    assert audit_exact_topk(prr_i, prr_s, orr_i, orr_s, tol=5e-3)["agreement"] == 1.0
    ps1024_i, ps1024_s, _ = idx_m3_gpu.profile_slice_pruning(qt_m3, qo_m3, topk=10, c_warm=16)
    assert audit_exact_topk(ps1024_i, ps1024_s, or_m3_i, or_m3_s, tol=5e-3)["agreement"] == 1.0
    print("[PASS] 13_d1024_fused_shm_pass16_and_rerank")

    # 14. Dynamic streaming insert_docs + delete_docs interleaved exactness
    init_n = 300
    init_offs = d_offs[:init_n + 1]
    init_toks = d_toks[:int(init_offs[-1])]
    dyn_idx = MaxBoundIndex(init_toks, init_offs, device=dev, mode="int8")
    dyn_idx.insert_docs(d_toks[int(d_offs[init_n]):int(d_offs[400])], d_offs[init_n:401] - d_offs[init_n])
    dyn_idx.insert_docs(d_toks[int(d_offs[400]):int(d_offs[500])], d_offs[400:501] - d_offs[400])
    del_set = sorted(set(int(x) for x in o_ids[:8, 0]).union(set(range(10, 40))))
    dyn_idx.delete_docs(del_set)
    keep_docs = [i for i in range(500) if i not in set(del_set)]
    keep_chunks = [d_toks[int(d_offs[i]):int(d_offs[i + 1])] for i in keep_docs]
    keep_lens = [c.shape[0] for c in keep_chunks]
    keep_offs = np.zeros(len(keep_docs) + 1, dtype=np.int64); np.cumsum(keep_lens, out=keep_offs[1:])
    keep_toks = torch.cat(keep_chunks, dim=0)
    ok_i_local, ok_s = exact_maxsim_oracle(q_toks.float(), q_offs, keep_toks, keep_offs, topk=10, device=dev, use_fp64=True)
    ok_i = np.array(keep_docs, dtype=np.int64)[ok_i_local]
    pd_i, pd_s, _ = dyn_idx.search(q_toks, q_offs, topk=10, c_warm=16)
    aud14 = audit_exact_topk(pd_i, pd_s, ok_i, ok_s, tol=2e-3)
    assert aud14["agreement"] == 1.0 and aud14["num_mismatch"] == 0, f"14_dynamic FAILED: {aud14}"
    print("[PASS] 14_streaming_insert_and_tombstone_delete")

    # 15. 4-Slice progressive pruning exactness
    ps_i, ps_s, sl_info = idx8.profile_slice_pruning(q_toks, q_offs, topk=10, c_warm=16)
    aud15 = audit_exact_topk(ps_i, ps_s, o_ids, o_sc, tol=2e-3)
    assert aud15["agreement"] == 1.0 and aud15["num_mismatch"] == 0, f"15_slice FAILED: {aud15}"
    print(f"[PASS] 15_progressive_4slice_pruning ({sl_info})")

    # 16. Strong Baseline B8_Fused_FP16_TC_Exact on D=128 (variable query lens + B=1,16) and D=1024!
    b8_128 = B8FusedFP16TCExact(d_toks, d_offs, device=dev)
    b8_i, b8_s = b8_128.search(q_toks, q_offs, topk=10)
    aud16a = audit_exact_topk(b8_i, b8_s, o_ids, o_sc, tol=2e-3)
    assert aud16a["agreement"] == 1.0 and aud16a["num_mismatch"] == 0, f"16a B8 D=128 FAILED: {aud16a}"
    o6_i, o6_s = exact_maxsim_oracle(qt6.float(), qo6, d_toks, d_offs, topk=10, device=dev, use_fp64=True)
    b8_i6, b8_s6 = b8_128.search(qt6, qo6, topk=10)
    aud16b = audit_exact_topk(b8_i6, b8_s6, o6_i, o6_s, tol=3e-3)
    assert aud16b["agreement"] == 1.0, f"16b B8 var_qlen FAILED: {aud16b}"
    b8_1024 = B8FusedFP16TCExact(dt_m3, offs_m3, device=dev)
    b8_im3, b8_sm3 = b8_1024.search(qt_m3, qo_m3, topk=10)
    aud16c = audit_exact_topk(b8_im3, b8_sm3, or_m3_i, or_m3_s, tol=5e-3)
    assert aud16c["agreement"] == 1.0, f"16c B8 D=1024 FAILED: {aud16c}"
    print("[PASS] 16_B8_fused_fp16_tc_exact_128d_and_1024d")

    # 17. Official FastPLAID GPU engine smoke & validity check
    fp_eng = B4OfficialFastPlaid(d_toks, d_offs, device=dev, nbits=2, kmeans_niters=2)
    fp_i, fp_s = fp_eng.search(q_toks, q_offs, n_ivf_probe=8, n_full_scores=256, topk=10)
    fp_eng.cleanup()
    assert fp_i.shape == (16, 10) and np.all(fp_i >= 0) and np.all(np.isfinite(fp_s))
    print("[PASS] 17_B4_official_fastplaid_gpu_verified")

    # 18. SIGIR'25 WARP GPU baseline smoke & high-probe convergence check
    warp_eng = B9WarpGPU(d_toks, d_offs, device=dev, n_centroids=64, kmeans_iters=4)
    w_i, w_s = warp_eng.search(q_toks, q_offs, t_prime=64, topk=10)
    aud18 = audit_exact_topk(w_i, w_s, o_ids, o_sc, tol=0.25)
    assert w_i.shape == (16, 10) and aud18["agreement"] >= 0.85, f"18_warp high-probe recall low: {aud18}"
    print(f"[PASS] 18_B9_warp_gpu_verified (full-probe agreement={aud18['agreement']:.3f})")

    # 19. Fused 7-stage FWHT-128 CUDA query prep bit-accuracy vs dense Hadamard H_t (128D & 1024D)
    for d_test, idx_test, qt_test, qo_test in [(128, idx8, q_toks, q_offs), (1024, idx_m3_gpu, qt_m3, qo_m3)]:
        _, m_mx, m_pd, ql_g, qf_p, _, _, _, _ = idx_test._prepare_queries(qt_test, qo_test, pad_mult=32, for_rerank=True)
        qi8_k = torch.empty((16, m_pd, d_test), dtype=torch.int8, device=dev)
        qsc_k = torch.empty((16, m_pd), dtype=torch.float32, device=dev)
        qnm_k = torch.empty((16, m_pd), dtype=torch.float32, device=dev)
        eps_k = torch.empty((16,), dtype=torch.float32, device=dev)
        idx_test.ext.prepare_queries_hadamard_int8(qf_p, ql_g, qi8_k, qsc_k, qnm_k, eps_k, idx_test.max_d_norm)
        q_rot_ref = qf_p.float() @ idx_test.H_t
        q_sc_ref = q_rot_ref.abs().amax(dim=2).clamp(min=1e-6) / 127.0
        max_sc_diff = float((qsc_k[:, :m_mx] - q_sc_ref).abs().max().item())
        max_nm_diff = float((qnm_k[:, :m_mx] - qf_p.float().norm(dim=2)).abs().max().item())
        assert max_sc_diff < 1e-5 and max_nm_diff < 1e-4, f"19_fwht D={d_test} diff too large: sc={max_sc_diff}, nm={max_nm_diff}"
    print("[PASS] 19_fwht128_cuda_query_prep_bit_accuracy_128d_and_1024d")

    # 20. ECIR'24 EMVB GPU baseline (Bit-Vector Popcount + PQ16 ADC + Exact Refinement + Mode-B Hybrid)
    emvb_eng = B10EmvbGPU(d_toks, d_offs, device=dev, n_centroids=64, m_pq=16, kmeans_iters=4)
    em_full_i, em_full_s = emvb_eng.search(q_toks, q_offs, nprobe=64, ncands=500, nrefine=500, topk=10)
    aud20_full = audit_exact_topk(em_full_i, em_full_s, o_ids, o_sc, tol=2e-3)
    assert aud20_full["agreement"] == 1.0, f"20_emvb full-probe FAILED: {aud20_full}"
    em_i, em_s = emvb_eng.search(q_toks, q_offs, nprobe=16, ncands=256, nrefine=128, topk=10)
    set_rec20 = float(np.mean([len(set(em_i[q]) & set(o_ids[q])) / 10.0 for q in range(16)]))
    assert em_i.shape == (16, 10) and set_rec20 >= 0.85, f"20_emvb set recall low: {set_rec20}"
    em_cands = emvb_eng.get_candidates(q_toks, q_offs, nprobe=16, ncands=128)
    or_em_i, or_em_s = exact_maxsim_oracle(q_toks.float(), q_offs, d_toks, d_offs, topk=10, device=dev, candidate_ids=em_cands.cpu().numpy(), use_fp64=True)
    rr_em_i, rr_em_s, _ = idx8.search_rerank(q_toks, q_offs, em_cands, topk=10, c_warm=16)
    assert audit_exact_topk(rr_em_i, rr_em_s, or_em_i, or_em_s, tol=2e-3)["agreement"] == 1.0
    print(f"[PASS] 20_B10_emvb_gpu_ecir24_and_modeB_verified (full_probe=1.000, approx_recall@10={set_rec20:.3f})")

    print("\nALL 20 ADVERSARIAL FP64-AUDITED EXACTNESS & BASELINE TESTS PASSED 100.000%!")


if __name__ == "__main__":
    main()
