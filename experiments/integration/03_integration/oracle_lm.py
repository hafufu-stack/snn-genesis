"""
03_integration: Oracle LM — SNN-ANN Hybrid Architecture
========================================================

From Paper 2 (Hybrid SNN-LM v4) & Paper 4 (AI Immune System v11):

The ultimate integration: a hybrid system where:
  - ANN (Mistral-7B) handles heavy language generation
  - SNN co-processor handles:
    * Canary detection (anomaly monitoring)
    * Chaos injection (data diversity)
    * Energy-efficient inference (BitNet ternary weights)

Architecture (The Neural Bridge):
  ┌─────────────────────────────┐
  │     ANN Backbone (LLM)      │
  │     Mistral-7B / Llama      │
  │                             │
  │  Layer 10 ──┐               │
  │             │ hidden state   │
  │             ▼               │
  │  ┌──────────────────┐      │
  │  │ SNN Co-Processor  │      │
  │  │                  │      │
  │  │  ┌────────────┐  │      │
  │  │  │ Canary SNN  │←─── entropy monitoring
  │  │  └────────────┘  │      │
  │  │  ┌────────────┐  │      │
  │  │  │ Chaos SNN   │───→ noise injection (if safe)
  │  │  └────────────┘  │      │
  │  │  ┌────────────┐  │      │
  │  │  │ BitNet Gate │←─── ternary weight efficiency
  │  │  └────────────┘  │      │
  │  └──────────────────┘      │
  │             │               │
  │  Layer 11 ──┘               │
  └─────────────────────────────┘
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
from core.snn_reservoir import ChaoticReservoir


class BitNetTernaryLayer(nn.Module):
    """
    BitNet b1.58 ternary weight layer from Hybrid SNN-LM (Paper 2).

    Weights are constrained to {-1, 0, +1}, eliminating multiplication:
      output = sum(input * weight) → just additions and subtractions

    Benefits:
      - ~14.7× less compute than FP16
      - ~5× less memory
      - Ideal for SNN co-processor (low power)
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def ternary_quantize(self, w):
        """Quantize weights to {-1, 0, +1} using threshold."""
        threshold = 0.7 * w.abs().mean()
        return torch.sign(w) * (w.abs() > threshold).float()

    def forward(self, x):
        w_ternary = self.ternary_quantize(self.weight)
        return nn.functional.linear(x, w_ternary)


class HybridReadout(nn.Module):
    """
    Hybrid spike + membrane readout from Paper 2.

    Standard SNN readouts lose information by only counting spikes.
    Hybrid readout combines:
      - Spike count (digital, discrete)
      - Membrane potential (analog, continuous)

    This gives +39.7% accuracy improvement over spike-only readout.
    """

    def __init__(self, hidden_dim, output_dim, spike_weight=0.6, membrane_weight=0.4):
        super().__init__()
        self.spike_layer = BitNetTernaryLayer(hidden_dim, output_dim)
        self.membrane_layer = BitNetTernaryLayer(hidden_dim, output_dim)
        self.spike_weight = spike_weight
        self.membrane_weight = membrane_weight

    def forward(self, spikes, membranes):
        spike_out = self.spike_layer(spikes)
        membrane_out = self.membrane_layer(membranes)
        return self.spike_weight * spike_out + self.membrane_weight * membrane_out


class CanarySNN(nn.Module):
    """
    SNN-based canary head monitor.

    Instead of computing attention entropy from the ANN,
    this SNN module natively detects anomalies in hidden states
    using spike-rate coding.

    Input: LLM hidden state from canary layer
    Output: anomaly score (high = nightmare detected)
    """

    def __init__(self, input_dim=4096, hidden_dim=128, threshold=1.0):
        super().__init__()
        self.encoder = BitNetTernaryLayer(input_dim, hidden_dim)
        self.readout = BitNetTernaryLayer(hidden_dim, 1)
        self.threshold = threshold

    def forward(self, hidden_state):
        # Encode hidden state to spiking representation
        rates = torch.sigmoid(self.encoder(hidden_state))

        # Spike generation (stochastic)
        spikes = (torch.rand_like(rates) < rates).float()

        # Anomaly detection via readout on spiking representation
        anomaly_score = torch.sigmoid(self.readout(spikes)).squeeze(-1)
        return anomaly_score


class ChaosSNN(nn.Module):
    """
    SNN-based chaos noise generator for the co-processor.

    Uses ChaoticReservoir from SNN-Comprypto to generate noise,
    but gates it through BitNet ternary weights for energy efficiency.
    """

    def __init__(self, output_dim=4096, reservoir_size=300, temperature=1.0):
        super().__init__()
        self.reservoir = ChaoticReservoir(
            num_neurons=reservoir_size,
            temperature=temperature,
            seed=2026
        )
        self.gate = BitNetTernaryLayer(reservoir_size, output_dim)
        self.output_dim = output_dim

    def forward(self, sigma=0.10):
        """Generate gated chaos noise."""
        noise_np = self.reservoir.generate_noise_vector(
            self.gate.in_features, warmup_steps=5
        )
        noise = torch.from_numpy(noise_np).float()
        gated = self.gate(noise) * sigma
        return gated


class OracleLM(nn.Module):
    """
    Oracle LM: SNN-ANN Hybrid Architecture.

    The co-processor sits at the canary layer and provides:
      1. Anomaly detection (canary SNN)
      2. Chaos noise injection (chaos SNN)
      3. Self-healing trigger (if anomaly detected)

    This is the architecture described in Deep Think's Proposal 2.
    """

    def __init__(self, input_dim=4096, canary_threshold=0.5):
        super().__init__()
        self.canary = CanarySNN(input_dim=input_dim)
        self.chaos = ChaosSNN(output_dim=input_dim)
        self.canary_threshold = canary_threshold

    def process_hidden_state(self, hidden_state, mode="monitor"):
        """
        Process a hidden state from the ANN backbone.

        Modes:
          - "monitor": Only detect anomalies (production)
          - "evolve":  Inject chaos noise for data generation (training)
          - "heal":    Detect + heal if anomalous (self-repair)
        """
        anomaly_score = self.canary(hidden_state)

        if mode == "monitor":
            is_anomaly = (anomaly_score > self.canary_threshold).any().item()
            return hidden_state, {
                "anomaly_score": anomaly_score.mean().item(),
                "is_anomaly": is_anomaly,
            }

        elif mode == "evolve":
            chaos_noise = self.chaos()
            chaos_noise = chaos_noise.to(device=hidden_state.device,
                                         dtype=hidden_state.dtype)
            # Shape broadcast
            if chaos_noise.shape != hidden_state.shape[-1:]:
                chaos_noise = chaos_noise[:hidden_state.shape[-1]]
            return hidden_state + chaos_noise, {
                "anomaly_score": anomaly_score.mean().item(),
                "noise_injected": True,
            }

        elif mode == "heal":
            is_anomaly = (anomaly_score > self.canary_threshold).any().item()
            if is_anomaly:
                # Dampen the hidden state to reduce anomaly
                scale = 1.0 - anomaly_score.unsqueeze(-1) * 0.1
                return hidden_state * scale, {
                    "anomaly_score": anomaly_score.mean().item(),
                    "healing_applied": True,
                }
            return hidden_state, {
                "anomaly_score": anomaly_score.mean().item(),
                "healing_applied": False,
            }


def demo():
    """Demonstrate Oracle LM co-processor."""
    print("=" * 60)
    print("Oracle LM: SNN-ANN Hybrid Co-Processor Demo")
    print("=" * 60)

    oracle = OracleLM(input_dim=256)  # Small for demo

    # Simulate hidden state
    normal_hs = torch.randn(1, 10, 256) * 0.5
    anomalous_hs = torch.randn(1, 10, 256) * 5.0  # Much larger magnitude

    print("\n1. Monitor Mode:")
    _, info = oracle.process_hidden_state(normal_hs, mode="monitor")
    print(f"   Normal hidden state:    anomaly={info['anomaly_score']:.4f}, "
          f"is_anomaly={info['is_anomaly']}")
    _, info = oracle.process_hidden_state(anomalous_hs, mode="monitor")
    print(f"   Anomalous hidden state: anomaly={info['anomaly_score']:.4f}, "
          f"is_anomaly={info['is_anomaly']}")

    print("\n2. Evolve Mode (Data Generation):")
    evolved, info = oracle.process_hidden_state(normal_hs, mode="evolve")
    diff = (evolved - normal_hs).norm().item()
    print(f"   Noise injected: {info['noise_injected']}")
    print(f"   Hidden state change: {diff:.4f}")

    print("\n3. Heal Mode (Self-Repair):")
    healed, info = oracle.process_hidden_state(anomalous_hs, mode="heal")
    print(f"   Healing applied: {info['healing_applied']}")

    # BitNet efficiency
    print(f"\n{'=' * 60}")
    print("BitNet Ternary Weight Efficiency")
    print("=" * 60)
    layer = BitNetTernaryLayer(256, 128)
    w = layer.weight.data
    w_ternary = layer.ternary_quantize(w)
    nonzero = (w_ternary != 0).float().mean().item()
    print(f"  Weight sparsity: {(1-nonzero)*100:.1f}%")
    print(f"  Active weights:  {nonzero*100:.1f}%")
    print(f"  Values: only {{-1, 0, +1}} → no multiplication needed!")
    print(f"  Estimated speedup: ~14.7× vs FP16")


if __name__ == "__main__":
    demo()
