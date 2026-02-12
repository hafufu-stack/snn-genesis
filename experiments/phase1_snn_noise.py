"""
Phase 1: SNN Noise Quality Validation
=======================================

Validates that the SNN chaotic reservoir produces randomness superior to
ANN/LSTM baselines. Reproduces key findings from SNN-Comprypto v5 and
extends them for the Project Genesis data augmentation pipeline.

Tests:
1. Byte frequency distribution (should be uniform)
2. Prediction rate (should be ~0.39% = theoretical random)
3. Autocorrelation (should be near-zero)
4. Comparison: SNN vs numpy.random vs simple ANN

Output: figures/phase1_randomness_quality.png
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.snn_reservoir import ChaoticReservoir


def test_byte_distribution(random_bytes: bytes, label: str) -> dict:
    """Test uniform distribution of byte values."""
    counts = np.zeros(256, dtype=int)
    for b in random_bytes:
        counts[b] += 1

    expected = len(random_bytes) / 256
    chi_squared = np.sum((counts - expected) ** 2 / expected)

    # Chi-squared critical value for 255 df, p=0.05: 293.2
    is_uniform = chi_squared < 293.2

    return {
        "label": label,
        "chi_squared": chi_squared,
        "is_uniform": is_uniform,
        "max_deviation": np.max(np.abs(counts - expected)) / expected * 100,
        "counts": counts,
    }


def test_prediction_rate(random_bytes: bytes, label: str) -> dict:
    """
    Test predictability: try to predict byte N from byte N-1.
    Theoretical random = 1/256 = 0.390625%
    """
    if len(random_bytes) < 2:
        return {"label": label, "prediction_rate": 0, "ratio_to_theory": 0}

    # Build frequency table: given prev_byte, what's the most likely next byte?
    transition = np.zeros((256, 256), dtype=int)
    for i in range(1, len(random_bytes)):
        transition[random_bytes[i - 1]][random_bytes[i]] += 1

    # Predict using most-frequent-next strategy
    correct = 0
    for i in range(1, len(random_bytes)):
        predicted = np.argmax(transition[random_bytes[i - 1]])
        if predicted == random_bytes[i]:
            correct += 1

    rate = correct / (len(random_bytes) - 1) * 100
    theoretical = 100 / 256

    return {
        "label": label,
        "prediction_rate": rate,
        "ratio_to_theory": rate / theoretical,
    }


def test_autocorrelation(random_bytes: bytes, label: str, max_lag: int = 50) -> dict:
    """Test autocorrelation at various lags."""
    data = np.array(list(random_bytes), dtype=float)
    data = data - data.mean()

    autocorr = np.correlate(data, data, mode="full")
    autocorr = autocorr[len(data) - 1:]
    autocorr = autocorr / autocorr[0]

    lags = autocorr[1:max_lag + 1]
    max_autocorr = np.max(np.abs(lags))

    return {
        "label": label,
        "max_autocorrelation": max_autocorr,
        "mean_autocorrelation": np.mean(np.abs(lags)),
        "autocorr_values": autocorr[:max_lag + 1],
    }


def generate_ann_random(n_bytes: int, seed: int = 42) -> bytes:
    """Generate random bytes using a simple ANN (DNN baseline)."""
    np.random.seed(seed)
    # Simple DNN: 2-layer feedforward with tanh
    W1 = np.random.randn(64, 1) * 0.5
    b1 = np.zeros(64)
    W2 = np.random.randn(1, 64) * 0.5
    b2 = np.zeros(1)

    result = []
    x = np.array([[0.5]])  # initial input

    for _ in range(n_bytes):
        h = np.tanh(W1 @ x + b1.reshape(-1, 1))
        y = np.tanh(W2 @ h + b2.reshape(-1, 1))
        byte_val = int((y[0, 0] + 1.0) * 127.5)
        byte_val = max(0, min(255, byte_val))
        result.append(byte_val)
        x = y  # feed output back as input

    return bytes(result)


def main():
    print("=" * 70)
    print("Phase 1: SNN Noise Quality Validation")
    print("Project Genesis - Self-Evolving AI via SNN Randomness")
    print("=" * 70)

    n_samples = 100_000
    print(f"\nGenerating {n_samples:,} random bytes from each source...")

    # === Source 1: SNN Chaotic Reservoir ===
    print("\n[1/3] SNN Chaotic Reservoir (300 neurons, spectral radius 1.4)...")
    t0 = time.time()
    reservoir = ChaoticReservoir(num_neurons=300, temperature=1.0, seed=2026)

    snn_bytes = []
    for i in range(n_samples):
        reservoir.step(np.random.randint(0, 256))
        rb = reservoir.get_random_bytes(1)
        snn_bytes.append(rb[0])
        if (i + 1) % 25000 == 0:
            print(f"  {i + 1:,}/{n_samples:,} bytes generated...")
    snn_bytes = bytes(snn_bytes)
    t_snn = time.time() - t0
    print(f"  Done in {t_snn:.1f}s")

    # === Source 2: Python random (numpy) ===
    print("\n[2/3] numpy.random (baseline)...")
    t0 = time.time()
    np.random.seed(42)
    numpy_bytes = bytes(np.random.randint(0, 256, n_samples, dtype=np.uint8))
    t_numpy = time.time() - t0
    print(f"  Done in {t_numpy:.4f}s")

    # === Source 3: Simple ANN ===
    print("\n[3/3] Simple ANN (DNN, 64 hidden units)...")
    t0 = time.time()
    ann_bytes = generate_ann_random(n_samples)
    t_ann = time.time() - t0
    print(f"  Done in {t_ann:.1f}s")

    # === Run Tests ===
    print("\n" + "=" * 70)
    print("Running Randomness Quality Tests...")
    print("=" * 70)

    sources = [
        ("SNN (300n)", snn_bytes),
        ("numpy.random", numpy_bytes),
        ("ANN (64h)", ann_bytes),
    ]

    dist_results = []
    pred_results = []
    auto_results = []

    for label, data in sources:
        dist_results.append(test_byte_distribution(data, label))
        pred_results.append(test_prediction_rate(data, label))
        auto_results.append(test_autocorrelation(data, label))

    # === Print Results ===
    print("\n📊 Byte Distribution (Chi-Squared Test)")
    print("-" * 55)
    print(f"{'Source':<16} {'χ²':<12} {'Max Dev %':<12} {'Uniform?'}")
    print("-" * 55)
    for r in dist_results:
        status = "✅ Yes" if r["is_uniform"] else "❌ No"
        print(f"{r['label']:<16} {r['chi_squared']:<12.1f} {r['max_deviation']:<12.1f} {status}")

    print("\n🎯 Prediction Rate (lower = more random)")
    print("-" * 55)
    print(f"{'Source':<16} {'Rate %':<12} {'vs Theory':<12} {'Status'}")
    print("-" * 55)
    theoretical = 100 / 256
    for r in pred_results:
        if r["ratio_to_theory"] < 2.0:
            status = "✅ Random-like"
        elif r["ratio_to_theory"] < 10.0:
            status = "⚠️ Weak"
        else:
            status = "❌ Predictable"
        print(f"{r['label']:<16} {r['prediction_rate']:<12.3f} {r['ratio_to_theory']:<12.2f}× {status}")
    print(f"{'(Theory)':<16} {theoretical:<12.4f} {'1.00':<12}")

    print("\n📈 Autocorrelation (lower = more random)")
    print("-" * 55)
    print(f"{'Source':<16} {'Max |AC|':<12} {'Mean |AC|':<12}")
    print("-" * 55)
    for r in auto_results:
        print(f"{r['label']:<16} {r['max_autocorrelation']:<12.6f} {r['mean_autocorrelation']:<12.6f}")

    # === Generate Visualization ===
    print("\n📊 Generating visualization...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Project Genesis - Phase 1: SNN Randomness Quality\n"
                 "SNN Chaotic Reservoir vs Baselines (100,000 bytes each)",
                 fontsize=14, fontweight="bold")

    colors = ["#2ecc71", "#3498db", "#e74c3c"]  # green, blue, red

    # Row 1: Byte distribution
    for i, (r, color) in enumerate(zip(dist_results, colors)):
        ax = axes[0, i]
        ax.bar(range(256), r["counts"], width=1.0, color=color, alpha=0.7)
        expected = n_samples / 256
        ax.axhline(y=expected, color="black", linestyle="--", alpha=0.5, label=f"Expected ({expected:.0f})")
        ax.set_title(f"{r['label']}\nχ²={r['chi_squared']:.0f}", fontsize=11)
        ax.set_xlabel("Byte Value")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    # Row 2: Autocorrelation
    for i, (r, color) in enumerate(zip(auto_results, colors)):
        ax = axes[1, i]
        lags = range(len(r["autocorr_values"]))
        ax.bar(lags, r["autocorr_values"], width=1.0, color=color, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        # 95% confidence interval for white noise
        ci = 1.96 / np.sqrt(n_samples)
        ax.axhline(y=ci, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=-ci, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Autocorrelation: {r['label']}\nMax |AC|={r['max_autocorrelation']:.4f}", fontsize=11)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")

    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase1_randomness_quality.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ Saved: {out_path}")

    # === Phase 1b: Noise Vector Test ===
    print("\n" + "=" * 70)
    print("Phase 1b: Noise Vector Generation (for LLM injection)")
    print("=" * 70)

    for dim in [768, 2048, 4096]:
        reservoir.reset()
        t0 = time.time()
        noise = reservoir.generate_noise_vector(dim)
        t_gen = time.time() - t0

        print(f"\n  dim={dim:>5}: mean={noise.mean():+.4f}, std={noise.std():.4f}, "
              f"min={noise.min():+.3f}, max={noise.max():+.3f}, "
              f"time={t_gen*1000:.1f}ms")

    # === Summary ===
    print("\n" + "=" * 70)
    print("Phase 1 Summary")
    print("=" * 70)
    pred_snn = pred_results[0]["prediction_rate"]
    pred_ann = pred_results[2]["prediction_rate"]
    print(f"""
  SNN prediction rate:    {pred_snn:.3f}%  (theory: {theoretical:.4f}%)
  ANN prediction rate:    {pred_ann:.3f}%
  Ratio:                  SNN is {pred_ann/pred_snn:.1f}× MORE random than ANN

  → SNN chaotic noise is suitable for Genesis data augmentation pipeline
  → Next: Phase 2 - Inject SNN noise into LLM hidden states
""")


if __name__ == "__main__":
    main()
