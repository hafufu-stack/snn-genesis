"""
02_augmentation: SNN Chaos-Driven Data Augmentation
====================================================

From Paper 1 (SNN-Comprypto) & Paper 4 (AI Immune System v10):
- Chaotic reservoir generates cryptographic-grade noise
- Temperature parameter controls chaos intensity
- Canary head entropy detects when augmentation reaches nightmares
- Surgical CoT heals nightmares into diverse but correct data

Integration with Genesis:
- Comprypto's temperature = Electric Dreams' sigma
- Canary-triggered augmentation auto-adjusts noise level
- Adaptive temperature creates maximally diverse training data
"""

import torch
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
from core.snn_reservoir import ChaoticReservoir

# Default canary head from v11
CANARY_LAYER = 10
CANARY_HEAD = 17
NIGHTMARE_THRESHOLD = 3.0


class AdaptiveTemperatureController:
    """
    Links SNN-Comprypto temperature to Electric Dreams sigma.

    The chaotic reservoir's temperature controls:
      - LOW temp → predictable noise → weak augmentation
      - HIGH temp → chaotic noise → diverse augmentation → nightmares
      - OPTIMAL temp → "Edge of Chaos" → maximally diverse yet learnable

    Uses canary entropy as feedback to find the optimal temperature.
    """

    def __init__(self, initial_temp=1.0, min_temp=0.1, max_temp=5.0,
                 step=0.1, target_entropy=2.8):
        self.temperature = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.step = step
        self.target_entropy = target_entropy  # Just below nightmare threshold
        self.history = []

    def update(self, canary_entropy):
        """
        Adjust temperature based on canary feedback.

        If entropy is too low → increase temperature (more chaos)
        If entropy is too high → decrease (approaching nightmare)
        Target: just below NIGHTMARE_THRESHOLD for maximum diversity
        """
        error = self.target_entropy - canary_entropy
        adjustment = self.step * np.sign(error) * min(abs(error), 1.0)
        self.temperature = np.clip(
            self.temperature + adjustment, self.min_temp, self.max_temp
        )
        self.history.append({
            "entropy": round(canary_entropy, 4),
            "temp": round(self.temperature, 4),
            "error": round(error, 4),
        })
        return self.temperature


class CanaryTriggeredAugmenter:
    """
    Combines Comprypto chaos with Canary detection for intelligent
    data augmentation.

    Pipeline:
      1. Generate noise from SNN chaotic reservoir at current temperature
      2. Inject into LLM hidden state
      3. Monitor canary entropy
      4. If entropy > threshold → nightmare → heal via CoT
      5. If entropy < target → increase temperature for more diversity
      6. Result: maximally diverse yet correct training data
    """

    def __init__(self, temperature=1.0, sigma=0.10, reservoir_size=300):
        self.controller = AdaptiveTemperatureController(initial_temp=temperature)
        self.reservoir = ChaoticReservoir(
            num_neurons=reservoir_size,
            temperature=temperature,
            seed=2026
        )
        self.sigma = sigma

    def make_noise_hook(self):
        """Create forward pre-hook for noise injection."""
        res = self.reservoir
        sigma = self.sigma

        def pre_hook(module, args):
            hs = args[0]
            total = 1
            for s in hs.shape:
                total *= s
            noise_np = res.generate_noise_vector(total, warmup_steps=5)
            noise = torch.from_numpy(noise_np).reshape(hs.shape)
            noise = noise.to(device=hs.device, dtype=hs.dtype) * sigma
            return (hs + noise,) + args[1:]

        return pre_hook

    def adapt_temperature(self, canary_entropy):
        """
        Update temperature based on canary feedback.
        Returns the new temperature.
        """
        new_temp = self.controller.update(canary_entropy)
        self.reservoir = ChaoticReservoir(
            num_neurons=self.reservoir.num_neurons,
            temperature=new_temp,
            seed=int(np.random.randint(10000))
        )
        return new_temp


class SNN_Comprypto_Augmenter:
    """
    Direct augmentation using Comprypto-style chaotic encoding.

    Instead of just adding noise, this compresses and reconstructs
    the data through the chaotic reservoir, producing augmented
    versions that preserve semantic information while introducing
    controlled variation.
    """

    def __init__(self, reservoir_size=300, temperature=1.0):
        self.reservoir = ChaoticReservoir(
            num_neurons=reservoir_size,
            temperature=temperature,
            seed=42
        )

    def chaotic_transform(self, embedding, strength=0.1):
        """
        Transform an embedding through chaotic dynamics.

        Steps:
          1. Generate chaotic mask from reservoir
          2. Apply selective perturbation (keeps structure, adds diversity)
          3. Re-normalize to preserve magnitude
        """
        total = 1
        for s in embedding.shape:
            total *= s

        # Generate chaotic mask
        chaos = self.reservoir.generate_noise_vector(total, warmup_steps=10)
        chaos = torch.from_numpy(chaos).reshape(embedding.shape)
        chaos = chaos.to(device=embedding.device, dtype=embedding.dtype)

        # Apply selective perturbation
        mask = (chaos.abs() > 0.5).float()  # Only perturb where chaos is strong
        perturbed = embedding + mask * chaos * strength

        # Re-normalize to preserve magnitude
        scale = embedding.norm() / (perturbed.norm() + 1e-8)
        return perturbed * scale


def demo():
    """Demonstrate adaptive temperature control."""
    print("=" * 60)
    print("Adaptive Temperature Controller Demo")
    print("(Comprypto Temperature ↔ Canary Entropy Feedback)")
    print("=" * 60)

    controller = AdaptiveTemperatureController(
        initial_temp=1.0,
        target_entropy=2.8
    )

    # Simulate canary entropy readings
    simulated_entropies = [2.2, 2.4, 2.6, 2.8, 3.2, 3.5, 3.0, 2.9, 2.8, 2.7]

    print(f"\n{'Step':<6} {'Entropy':<10} {'Temperature':<12} {'Error':<8}")
    print("-" * 36)
    for i, entropy in enumerate(simulated_entropies):
        temp = controller.update(entropy)
        print(f"{i:<6} {entropy:<10.2f} {temp:<12.4f} {controller.history[-1]['error']:<8.4f}")

    print(f"\n→ Temperature converges to maintain entropy near {controller.target_entropy}")
    print(f"  Final temperature: {controller.temperature:.4f}")


if __name__ == "__main__":
    demo()
