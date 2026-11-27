import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/maxbound"))
from bench.common import write_result_json


def bench_gpu(dev_id):
    torch.cuda.set_device(dev_id)
    props = torch.cuda.get_device_properties(dev_id)
    # 1. HBM Bandwidth (vector copy 512 MB)
    n_elems = 128 * 1024 * 1024  # 512 MB float32
    a = torch.empty(n_elems, dtype=torch.float32, device=dev_id)
    b = torch.empty(n_elems, dtype=torch.float32, device=dev_id)
    a.fill_(1.0)
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize(dev_id)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(10):
        s.record()
        b.copy_(a)
        e.record()
        torch.cuda.synchronize(dev_id)
        times.append(s.elapsed_time(e))
    med_ms = float(np.median(times))
    hbm_bw_gbs = (2.0 * n_elems * 4 / 1e9) / (med_ms / 1e3)
    del a, b

    # 2. FP16 TensorCore GEMM TFLOPS (4096 x 4096 x 4096)
    M = N = K = 4096
    x = torch.randn(M, K, dtype=torch.float16, device=dev_id)
    y = torch.randn(K, N, dtype=torch.float16, device=dev_id)
    for _ in range(3):
        z = x @ y
    torch.cuda.synchronize(dev_id)
    times_g = []
    for _ in range(10):
        s.record()
        z = x @ y
        e.record()
        torch.cuda.synchronize(dev_id)
        times_g.append(s.elapsed_time(e))
    med_g_ms = float(np.median(times_g))
    tflops_fp16 = (2.0 * M * N * K / 1e12) / (med_g_ms / 1e3)
    del x, y, z

    # 3. Pinned H2D & D2H Bandwidth (512 MB)
    h_pin = torch.empty(n_elems, dtype=torch.float32, pin_memory=True)
    h_pin.fill_(2.0)
    d_buf = torch.empty(n_elems, dtype=torch.float32, device=dev_id)
    for _ in range(2):
        d_buf.copy_(h_pin, non_blocking=False)
    torch.cuda.synchronize(dev_id)
    t_h2d, t_d2h = [], []
    for _ in range(5):
        s.record()
        d_buf.copy_(h_pin, non_blocking=False)
        e.record()
        torch.cuda.synchronize(dev_id)
        t_h2d.append(s.elapsed_time(e))
        s.record()
        h_pin.copy_(d_buf, non_blocking=False)
        e.record()
        torch.cuda.synchronize(dev_id)
        t_d2h.append(s.elapsed_time(e))
    h2d_gbs = (n_elems * 4 / 1e9) / (float(np.median(t_h2d)) / 1e3)
    d2h_gbs = (n_elems * 4 / 1e9) / (float(np.median(t_d2h)) / 1e3)
    del h_pin, d_buf
    torch.cuda.empty_cache()
    return {
        "device_id": dev_id,
        "name": props.name,
        "sm": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / (1024**3), 2),
        "sm_count": props.multi_processor_count,
        "hbm_bw_gbs": round(hbm_bw_gbs, 2),
        "fp16_tflops": round(tflops_fp16, 2),
        "pcie_h2d_gbs": round(h2d_gbs, 2),
        "pcie_d2h_gbs": round(d2h_gbs, 2),
    }


def bench_cross_gpu():
    can_p2p_01 = torch.cuda.can_device_access_peer(0, 1)
    can_p2p_10 = torch.cuda.can_device_access_peer(1, 0)
    latencies = {}
    for b in [1, 16, 64, 256, 1024]:
        src = torch.ones(b, dtype=torch.float32, device=1)
        pin = torch.empty(b, dtype=torch.float32, pin_memory=True)
        dst = torch.empty(b, dtype=torch.float32, device=0)
        for _ in range(5):
            pin.copy_(src)
            dst.copy_(pin)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        ts = []
        for _ in range(50):
            t0 = time.perf_counter()
            pin.copy_(src)
            torch.cuda.synchronize(1)
            dst.copy_(pin)
            torch.cuda.synchronize(0)
            ts.append((time.perf_counter() - t0) * 1e6)
        latencies[f"B_{b}_us"] = round(float(np.median(ts)), 2)
    return {
        "can_p2p_0_to_1": bool(can_p2p_01),
        "can_p2p_1_to_0": bool(can_p2p_10),
        "tau_pinned_transfer_us": latencies,
    }


if __name__ == "__main__":
    recs = {
        "gpu0": bench_gpu(0),
        "gpu1": bench_gpu(1),
        "cross_gpu": bench_cross_gpu(),
    }
    print(json.dumps(recs, indent=2))
    write_result_json(os.path.expanduser("~/maxbound/results/p0_hardware.json"), [recs])
