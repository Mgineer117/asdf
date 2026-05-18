"""
Benchmark: mp.Process (fork) serial encoding vs VectorizedSampler batched encoding.

This script simulates the Atari training pipeline to measure overhead from:
  1. Process spawn/join
  2. mp.Queue serialization of numpy arrays
  3. Per-frame serial CNN encoding (CPU) vs batched encoding (GPU/CPU)
  4. Env step simulation latency (mimics ALE's ~0.5ms/step)

Run:  python tests/bench_sampler.py

Key insight: On the SLURM cluster (CUDA GPU), fork-based workers CANNOT use
the GPU for encoding because CUDA can't be re-initialized after fork().
So fork workers are stuck on CPU, while the vectorized sampler can use the GPU.
"""

import time
from math import ceil
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn


# ── Minimal CNN encoder (mirrors Nature-DQN backbone) ──────────────────────
class _SimpleEncoder(nn.Module):
    def __init__(self, input_chw=(1, 210, 160), encoder_dim=256):
        super().__init__()
        C, H, W = input_chw
        self.encoder_dim = encoder_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            flat_dim = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(flat_dim, encoder_dim)

    def forward(self, x):
        return self.fc(self.cnn(x))


# ── Simulate env.step() latency (ALE runs at ~0.3-0.5ms/step) ─────────────
ENV_STEP_MS = 0.4


def _sim_env_step(frame_shape):
    """Simulate ALE env.step(): produce a new frame + small delay."""
    time.sleep(ENV_STEP_MS / 1000.0)
    return np.random.randint(0, 256, size=frame_shape, dtype=np.uint8)


# ── Per-frame encode (like ArcadeWrapper._encode) ─────────────────────────
def _encode_single(encoder, raw_frame: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(raw_frame.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    with torch.no_grad():
        feat = encoder(t)
    return feat.squeeze(0).numpy()


# ── Batched encode (like VectorizedSampler._batch_encode) ─────────────────
def _encode_batch(encoder, raw_frames: np.ndarray, device="cpu") -> np.ndarray:
    t = torch.from_numpy(raw_frames.astype(np.float32) / 255.0)
    t = t.unsqueeze(1).to(device)  # (N, 1, H, W)
    with torch.no_grad():
        feat = encoder(t)
    return feat.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 1: mp.Process workers (current OnlineSampler)
# ═══════════════════════════════════════════════════════════════════════════
def _worker(pid, queue, encoder, n_steps, frame_shape):
    """Each worker: step env serially → encode each frame on CPU → collect."""
    encoded = np.zeros((n_steps, encoder.encoder_dim), dtype=np.float32)
    for i in range(n_steps):
        raw = _sim_env_step(frame_shape)
        encoded[i] = _encode_single(encoder, raw)
    queue.put((pid, encoded))


def bench_fork(encoder, total_steps, num_workers, frame_shape):
    steps_per_worker = ceil(total_steps / num_workers)
    t0 = time.time()

    # Spawn
    t_s = time.time()
    procs, queue = [], mp.Queue()
    for w in range(num_workers):
        p = mp.Process(target=_worker, args=(w, queue, encoder, steps_per_worker, frame_shape))
        procs.append(p)
        p.start()
    t_spawn = time.time() - t_s

    # Collect
    t_c = time.time()
    results = [None] * num_workers
    collected = 0
    while collected < num_workers:
        try:
            pid, data = queue.get(timeout=120)
            results[pid] = data
            collected += 1
        except Empty:
            break
    t_collect = time.time() - t_c

    # Join
    t_j = time.time()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()
        p.close()
    queue.close()
    t_join = time.time() - t_j

    total = time.time() - t0
    eff = sum(r.shape[0] for r in results if r is not None)
    return {
        "total": total, "spawn": t_spawn, "collect": t_collect, "join": t_join,
        "steps": eff,
    }


def _worker_raw(pid, queue, n_steps, frame_shape):
    """Each worker: step env serially → collect raw frames."""
    raw_frames = np.zeros((n_steps, *frame_shape), dtype=np.uint8)
    for i in range(n_steps):
        raw_frames[i] = _sim_env_step(frame_shape)
    queue.put((pid, raw_frames))


def bench_fork_batch(encoder, total_steps, num_workers, frame_shape, device="cpu"):
    steps_per_worker = ceil(total_steps / num_workers)
    
    enc = encoder.to(device)
    dummy = torch.zeros(1, 1, *frame_shape, device=device)
    traced = torch.jit.trace(enc, dummy)
    
    t0 = time.time()

    # Spawn
    t_s = time.time()
    procs, queue = [], mp.Queue()
    for w in range(num_workers):
        p = mp.Process(target=_worker_raw, args=(w, queue, steps_per_worker, frame_shape))
        procs.append(p)
        p.start()
    t_spawn = time.time() - t_s

    # Collect and Encode
    t_c = time.time()
    results = [None] * num_workers
    collected = 0
    while collected < num_workers:
        try:
            pid, data = queue.get(timeout=120)
            results[pid] = data
            collected += 1
        except Empty:
            break
            
    t_collect_raw = time.time() - t_c
    
    # We now have results (num_workers, steps_per_worker, H, W). We need to batch encode it.
    t_e = time.time()
    all_raw = np.concatenate(results, axis=0) # (total_steps, H, W)
    
    # Batch encode on main thread
    all_enc = np.zeros((total_steps, encoder.encoder_dim), dtype=np.float32)
    step = 0
    batch_size = 256
    while step < total_steps:
        n = min(batch_size, total_steps - step)
        t = torch.from_numpy(all_raw[step:step+n].astype(np.float32) / 255.0).unsqueeze(1).to(device)
        with torch.no_grad():
            feat = traced(t)
        all_enc[step:step+n] = feat.cpu().numpy()
        step += n
    t_encode = time.time() - t_e

    # Join
    t_j = time.time()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
            p.join()
        p.close()
    queue.close()
    t_join = time.time() - t_j

    total = time.time() - t0
    return {
        "total": total, "spawn": t_spawn, "collect": t_collect_raw, "encode": t_encode, "join": t_join,
        "steps": len(all_raw),
    }


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 2: Single-process vectorized + batched encoding
# ═══════════════════════════════════════════════════════════════════════════
def bench_vectorized(encoder, total_steps, batch_size, frame_shape, device="cpu"):
    """Single process: batch N env.step() + batch encode on device."""
    enc = encoder.to(device)
    t0 = time.time()

    all_enc = np.zeros((total_steps, enc.encoder_dim), dtype=np.float32)
    step = 0
    while step < total_steps:
        n = min(batch_size, total_steps - step)

        # Simulate parallel AsyncVectorEnv step() calls
        time.sleep(ENV_STEP_MS / 1000.0)
        raw = np.random.randint(0, 256, (n, *frame_shape), dtype=np.uint8)

        # Batch encode
        all_enc[step:step + n] = _encode_batch(enc, raw, device=device)
        step += n

    total = time.time() - t0
    return {"total": total, "steps": step}


# ═══════════════════════════════════════════════════════════════════════════
# APPROACH 3: Vectorized + JIT-traced encoder
# ═══════════════════════════════════════════════════════════════════════════
def bench_vectorized_jit(encoder, total_steps, batch_size, frame_shape, device="cpu"):
    enc = encoder.to(device)
    dummy = torch.zeros(1, 1, *frame_shape, device=device)
    traced = torch.jit.trace(enc, dummy)
    traced.encoder_dim = enc.encoder_dim

    t0 = time.time()
    all_enc = np.zeros((total_steps, enc.encoder_dim), dtype=np.float32)
    step = 0
    while step < total_steps:
        n = min(batch_size, total_steps - step)
        time.sleep(ENV_STEP_MS / 1000.0)
        raw = np.random.randint(0, 256, (n, *frame_shape), dtype=np.uint8)
        t = torch.from_numpy(raw.astype(np.float32) / 255.0).unsqueeze(1).to(device)
        with torch.no_grad():
            feat = traced(t)
        all_enc[step:step + n] = feat.cpu().numpy()
        step += n

    total = time.time() - t0
    return {"total": total, "steps": step}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

    FRAME_SHAPE = (210, 160)
    TOTAL_STEPS = 2048         # realistic: ~1 on-policy batch
    NUM_WORKERS = 4
    ENCODER_DIM = 256

    print("=" * 70)
    print("BENCHMARK: mp.Process (fork) vs Vectorized (batched) Sampling")
    print("=" * 70)
    print(f"  Frame shape:    {FRAME_SHAPE}")
    print(f"  Total steps:    {TOTAL_STEPS}")
    print(f"  Num workers:    {NUM_WORKERS}")
    print(f"  Encoder dim:    {ENCODER_DIM}")
    print(f"  Env step sim:   {ENV_STEP_MS}ms per step")

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"  Device:         {device}")
    print()

    encoder = _SimpleEncoder(input_chw=(1, *FRAME_SHAPE), encoder_dim=ENCODER_DIM)
    encoder.eval()

    # Warmup
    print("Warming up...")
    _encode_single(encoder.cpu(), np.random.randint(0, 256, FRAME_SHAPE, dtype=np.uint8))
    _encode_batch(encoder.cpu(), np.random.randint(0, 256, (4,) + FRAME_SHAPE, dtype=np.uint8))
    print()

    # ── [1] Fork ──
    print("─" * 70)
    print("[1] mp.Process fork: 4 workers × serial (env.step + encode) on CPU")
    print("    (Workers CANNOT use GPU after fork — this is the key problem)")
    print("─" * 70)
    encoder_cpu = encoder.cpu()
    r1 = bench_fork(encoder_cpu, TOTAL_STEPS, NUM_WORKERS, FRAME_SHAPE)
    print(f"  Total time:          {r1['total']:.3f}s")
    print(f"    ├─ Process spawn:  {r1['spawn']:.3f}s")
    print(f"    ├─ Queue collect:  {r1['collect']:.3f}s  (includes all worker computation)")
    print(f"    └─ Process join:   {r1['join']:.3f}s")
    print(f"  Steps collected:     {r1['steps']}")
    sps1 = r1["steps"] / r1["total"]
    print(f"  Steps/sec:           {sps1:.0f}")
    print()

    # ── [1.5] Fork + Batch ──
    print("─" * 70)
    print("[2] mp.Process fork + batched encoding (CPU)")
    print("    (Shows overhead of passing large raw frames via mp.Queue)")
    print("─" * 70)
    r1b = bench_fork_batch(encoder_cpu, TOTAL_STEPS, NUM_WORKERS, FRAME_SHAPE, device="cpu")
    print(f"  Total time:          {r1b['total']:.3f}s")
    print(f"    ├─ Process spawn:  {r1b['spawn']:.3f}s")
    print(f"    ├─ Queue collect:  {r1b['collect']:.3f}s  (raw pixels)")
    print(f"    ├─ Main batch enc: {r1b['encode']:.3f}s")
    print(f"    └─ Process join:   {r1b['join']:.3f}s")
    sps1b = r1b["steps"] / r1b["total"]
    print(f"  Steps/sec:           {sps1b:.0f}")
    print()

    # ── [2] Vectorized CPU ──
    print("─" * 70)
    print("[3] Vectorized + batch encode (CPU only)")
    print("─" * 70)
    r2 = bench_vectorized(encoder_cpu, TOTAL_STEPS, NUM_WORKERS, FRAME_SHAPE, device="cpu")
    sps2 = r2["steps"] / r2["total"]
    print(f"  Total time:          {r2['total']:.3f}s")
    print(f"  Steps/sec:           {sps2:.0f}")
    print()

    # ── [3] Vectorized + JIT CPU ──
    print("─" * 70)
    print("[4] Vectorized + batch + JIT-traced (CPU)")
    print("─" * 70)
    r3 = bench_vectorized_jit(encoder_cpu, TOTAL_STEPS, NUM_WORKERS, FRAME_SHAPE, device="cpu")
    sps3 = r3["steps"] / r3["total"]
    print(f"  Total time:          {r3['total']:.3f}s")
    print(f"  Steps/sec:           {sps3:.0f}")
    print()

    # ── [4] Vectorized + JIT GPU ──
    r4 = None
    if device != "cpu":
        print("─" * 70)
        print(f"[5] Vectorized + batch + JIT-traced ({device.upper()})")
        print(f"    (On SLURM cluster, this is CUDA — the real speedup)")
        print("─" * 70)
        encoder_gpu = encoder.to(device)
        r4 = bench_vectorized_jit(encoder_gpu, TOTAL_STEPS, NUM_WORKERS, FRAME_SHAPE, device=device)
        sps4 = r4["steps"] / r4["total"]
        print(f"  Total time:          {r4['total']:.3f}s")
        print(f"  Steps/sec:           {sps4:.0f}")
        print()

    # ── Summary ──
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  {'Method':<50} {'Steps/s':>8} {'Speedup':>8}")
    print(f"  {'─' * 50} {'─' * 8} {'─' * 8}")
    print(f"  {'[1] Fork + serial CPU encode (baseline)':<50} {sps1:>8.0f} {'1.0x':>8}")
    print(f"  {'[2] Fork + batch encode (CPU)':<50} {sps1b:>8.0f} {sps1b/sps1:>7.1f}x")
    print(f"  {'[3] Vectorized + batch CPU encode':<50} {sps2:>8.0f} {sps2/sps1:>7.1f}x")
    print(f"  {'[4] Vectorized + batch + JIT (CPU)':<50} {sps3:>8.0f} {sps3/sps1:>7.1f}x")
    if r4:
        print(f"  {'[5] Vectorized + batch + JIT (' + device.upper() + ')':<50} {sps4:>8.0f} {sps4/sps1:>7.1f}x")
    print()

    print("Fork overhead breakdown:")
    t = r1["total"]
    print(f"  Process spawn:       {r1['spawn']:.3f}s  ({r1['spawn']/t*100:4.1f}%)")
    print(f"  Queue + compute:     {r1['collect']:.3f}s  ({r1['collect']/t*100:4.1f}%)")
    print(f"  Process join:        {r1['join']:.3f}s  ({r1['join']/t*100:4.1f}%)")
    print()

    print("KEY TAKEAWAY:")
    print("  On your Mac (MPS), fork may appear competitive because:")
    print("    • 4 workers run env.step() in TRUE parallel (separate processes)")
    print("    • MPS has high overhead for small batches")
    print()
    print("  On the SLURM cluster (CUDA), fork LOSES because:")
    print("    • Workers cannot use CUDA after fork — stuck on CPU for encoding")
    print("    • Queue serializes large numpy arrays (O(batch × encoder_dim))")
    print("    • VectorizedSampler batches encoding on GPU: 1 call for N frames")
    print("    • Expected speedup on CUDA: 5-20× for the encoding step")
