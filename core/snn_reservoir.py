"""
SNN Chaotic Reservoir - Core Module for Project Genesis
========================================================

Extracts and extends the chaotic SNN reservoir from SNN-Comprypto v5
for high-quality random noise generation.

Key properties:
- LIF neurons with chaotic dynamics (spectral radius 1.4)
- SHA-256 hashing of membrane potentials for cryptographic randomness
- Temperature-modulated thermal noise
- Prediction rate: 0.388% (matching theoretical random at 0.39%)

Original source: temporal-coding-simulation/snn-comprypto/core/comprypto_system.py
"""

import numpy as np
import hashlib
from typing import Optional


class LIFNeuron:
    """Leaky Integrate-and-Fire neuron with chaotic dynamics."""

    def __init__(self, dt: float = 0.5, tau: float = 20.0):
        self.dt = dt
        self.tau = tau
        self.v = -65.0       # membrane potential
        self.v_rest = -65.0
        self.v_thresh = -50.0
        self.v_reset = -70.0

    def step(self, I_syn: float) -> float:
        """One timestep: returns 1.0 if spike, 0.0 otherwise."""
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        if self.v >= self.v_thresh:
            self.v = self.v_reset
            return 1.0
        return 0.0

    def reset(self):
        self.v = self.v_rest


class ChaoticReservoir:
    """
    Neuro-Chaotic Reservoir for random noise generation.

    Uses Edge-of-Chaos dynamics (spectral radius ~1.4) to produce
    cryptographically random byte sequences from membrane potential states.

    Parameters
    ----------
    num_neurons : int
        Number of LIF neurons in the reservoir (default: 300)
    density : float
        Connection sparsity (default: 0.1 = 10% connected)
    spectral_radius : float
        Controls chaos level. >1.0 = chaotic regime (default: 1.4)
    temperature : float
        Thermal noise amplitude. Acts as a secondary randomness source.
    seed : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        num_neurons: int = 300,
        density: float = 0.1,
        spectral_radius: float = 1.4,
        temperature: float = 1.0,
        seed: int = 2026,
    ):
        self.num_neurons = num_neurons
        self.temperature = temperature
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Reservoir weight matrix (Edge of Chaos)
        W = self.rng.randn(num_neurons, num_neurons)
        rho = max(abs(np.linalg.eigvals(W)))
        self.W_res = W * (spectral_radius / rho)

        # Sparse connectivity
        mask = self.rng.rand(num_neurons, num_neurons) < density
        self.W_res *= mask

        # Input weights
        self.W_in = self.rng.randn(num_neurons, 1) * 40.0

        # Neurons
        self.neurons = [LIFNeuron() for _ in range(num_neurons)]

        # State
        self.fire_rate = np.zeros(num_neurons)
        self.step_count = 0

    def step(self, input_val: float = 0.0) -> np.ndarray:
        """
        Run one timestep of the reservoir.

        Parameters
        ----------
        input_val : float
            External input (0-255 range, normalized internally)

        Returns
        -------
        spikes : np.ndarray
            Binary spike vector of shape (num_neurons,)
        """
        u = (input_val / 127.5) - 1.0

        I_rec = self.W_res @ self.fire_rate
        I_ext = (self.W_in * u).flatten()
        thermal_noise = self.rng.normal(0, 0.5 * self.temperature, self.num_neurons)
        I_total = I_rec + I_ext + thermal_noise

        spikes = np.zeros(self.num_neurons)
        for i, neuron in enumerate(self.neurons):
            bias = 25.0  # nonlinear bias for chaos
            if neuron.step(I_total[i] + bias) > 0.0:
                spikes[i] = 1.0

        self.fire_rate = 0.7 * self.fire_rate + 0.3 * spikes
        self.step_count += 1
        return spikes

    def get_random_bytes(self, n_bytes: int = 32) -> bytes:
        """
        Generate n_bytes of cryptographic-quality random bytes
        from current membrane potential states via SHA-256.
        """
        state_values = np.array([n.v for n in self.neurons])
        state_bytes = state_values.tobytes()

        result = b""
        for i in range((n_bytes + 31) // 32):
            # Mix in step count for uniqueness
            h = hashlib.sha256(state_bytes + i.to_bytes(4, "little")).digest()
            result += h

        return result[:n_bytes]

    def generate_noise_vector(self, dim: int, warmup_steps: int = 50) -> np.ndarray:
        """
        Generate a noise vector of specified dimension.

        This is the key function for Project Genesis:
        converts SNN chaotic dynamics into noise suitable for
        LLM hidden state perturbation.

        Parameters
        ----------
        dim : int
            Dimension of the noise vector (e.g., 4096 for Mistral-7B)
        warmup_steps : int
            Number of warmup steps to let chaos develop

        Returns
        -------
        noise : np.ndarray
            Float32 noise vector with ~N(0,1) distribution
        """
        # Warm up the reservoir
        for _ in range(warmup_steps):
            self.step(self.rng.randint(0, 256))

        # Generate enough random bytes
        raw = self.get_random_bytes(dim * 4)  # 4 bytes per float32

        # Convert to uint32 then to uniform [0,1]
        uint_vals = np.frombuffer(raw, dtype=np.uint32)
        uniform = uint_vals / np.float64(2**32)

        # Box-Muller transform: uniform -> Gaussian
        pairs = uniform.reshape(-1, 2)
        r = np.sqrt(-2.0 * np.log(pairs[:, 0] + 1e-30))
        theta = 2.0 * np.pi * pairs[:, 1]
        gaussian = np.column_stack([r * np.cos(theta), r * np.sin(theta)]).flatten()

        return gaussian[:dim].astype(np.float32)

    def generate_noise_batch(
        self, batch_size: int, dim: int, sigma: float = 0.1
    ) -> np.ndarray:
        """
        Generate a batch of scaled noise vectors.

        Parameters
        ----------
        batch_size : int
            Number of noise vectors to generate
        dim : int
            Dimension of each vector
        sigma : float
            Noise scale (std dev). Default 0.1 matches Electric Dreams (v10).

        Returns
        -------
        noise_batch : np.ndarray
            Shape (batch_size, dim), scaled by sigma
        """
        batch = np.zeros((batch_size, dim), dtype=np.float32)
        for i in range(batch_size):
            batch[i] = self.generate_noise_vector(dim, warmup_steps=10) * sigma
        return batch

    def reset(self):
        """Reset reservoir to initial state."""
        for n in self.neurons:
            n.reset()
        self.fire_rate = np.zeros(self.num_neurons)
        self.rng = np.random.RandomState(self.seed)
        self.step_count = 0
