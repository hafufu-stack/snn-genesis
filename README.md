# 🧬 SNN-Genesis v10: Stochastic Resonance in LLM Reasoning — Orthogonal Noise Decomposition Reveals Direction×Magnitude Interaction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18625621.svg)](https://doi.org/10.5281/zenodo.18625621)

> **"Neither direction nor magnitude alone is sufficient. Stochastic resonance requires both."**

SNN-Genesis is a framework for LLM safety training and **interpretability** using biologically-inspired Spiking Neural Network (SNN) perturbations controlled by Closed-form Continuous-time (CfC) neural networks. v10 discovers **stochastic resonance in LLM reasoning**: additive noise at σ=0.15 boosts Modified Tower of Hanoi solve rate from 9% to 32% (p=8.4×10⁻⁵, N=100).

### 🆕 v10 Highlights (February 2026)

Season 11 discovers **stochastic resonance** in LLM reasoning and completes the **orthogonal noise decomposition**.

| Discovery | Key Result |
|-----------|------------|
| 🔔 **Stochastic Resonance** | Additive σ=0.15 at L18: 9% → **32%** solve rate (p=8.4×10⁻⁵, N=100) — 3.6× improvement |
| 🧩 **Orthogonal Decomposition** | Spherical (direction-only): ✗ / Radial (magnitude-only): ✗ / Additive (both): ✓ |
| 📈 **Bell Curve + Cliff** | Perfect inverted-U: peak at σ=0.15 (32%), complete collapse at σ=0.20 (0%) |
| 🔬 **Large-N Replication** | Peak replicated at N=100, cliff confirmed at N=100+50 |

### 📊 Retained from v9.1

| Discovery | Key Result |
|-----------|------------|
| 🍎 **Apple Rebuttal** | Chat template formatting transforms Hanoi: 0% → 16–28%. Apple's claim is a methodological artifact |
| 🔄 **Self-Correction is Real** | ~11 self-corrections/game across 170 trials |
| ❌ **SNN ≠ Reasoning at σ=0.02** | p=0.306, n.s. — now explained as sub-threshold dosing |
| 🧠 **DTR Ceiling Effect** | ~94.5% uniformly — success depends on *direction*, not *depth* |

### 📊 Retained from v8

| Discovery | Key Result |
|-----------|------------|
| 🌍 **Universal Homeostasis** | σ̄ = 0.074 ± 0.002 across Mistral, Qwen, Phi-3 — CV = 3.1% |
| 🧠 **CfC Brain Atlas** | Sharp specialization (Mistral) vs. distributed representation (Qwen) |
| 🛡️ **Plateau Robustness** | Qwen: 63%±8% Math from σ=0 to σ=0.15 (noise immunity) |
| 🔪 **Digital Lobotomy** | Causal proof: Δ=-1.7% (Qwen) vs. Δ=-5.0% (Mistral) |

### 📊 Benchmark Summary (v3.1)

| Benchmark | Mistral-7B Tax (σ=0.01) | Qwen2.5-7B Tax (σ=0.01) |
|-----------|------------------------|--------------------------| 
| TruthfulQA MC1 (817Q) | **-0.6%** | **-0.1%** |
| MMLU (1,600Q × 8 subj) | **0.0%** | **0.0%** |
| MMLU Full (14,042Q × 57 subj) | **0.00%** ✅ | — |
| HellaSwag (σ=0.01, L15-20) | **-0.70%** ✅ | — |
| ARC-Easy (σ=0.01, L15-20) | **-0.30%** ✅ | — |

## 🌌 The Vision: Three Phases of Artificial Brain Creation

```
 Phase 1: DREAM          Phase 2: EVOLUTION        Phase 3: TRANSCENDENCE
 ┌──────────────┐        ┌──────────────┐          ┌──────────────┐
 │  SNN Chaos   │        │  ANN Self-   │          │  Lossless    │
 │  Reservoir   │──────▶ │  Improvement │────────▶ │  Neural      │
 │  (Genesis)   │        │  (Evolution) │          │  Computing   │
 └──────────────┘        └──────────────┘          └──────────────┘
  SNN creates data        Data heals the ANN        Hybrid SNN-ANN
  (Chaos as data)         (Canary + Morpheus)        Brain vs Neumann
```

- **Phase 1 — Dream:** SNN chaotic dynamics generate infinite, cryptographic-grade synthetic training data. The temperature parameter controls diversity. Data scarcity is solved.
- **Phase 2 — Evolution:** The generated data fuels autonomous self-improvement. Canary heads detect hallucinations, dreams expose vulnerabilities, and QLoRA heals them. The model evolves its own immune system.
- **Phase 3 — Transcendence:** The SNN and ANN merge into a hybrid computing architecture. SNNs handle temporal processing, ANNs handle static reasoning. A fundamentally new computing paradigm.

## 📋 Version History

### v10 — Stochastic Resonance in LLM Reasoning (NEW)

**Season 11 (Phases 55-57): Noise Decomposition & Bell Curve**
- Phase 55: **Spherical noise** (direction-only) has no effect (12%, p=0.77 n.s.)
- Phase 55: **Additive noise** σ=0.10 → 38% (p=0.023 ★)
- Phase 56: **Radial noise** (magnitude-only) has no effect (4-12%, all n.s.)
- Phase 56: **Additive σ=0.15** → 38% (**p=0.0006 ★★★**, N=50)
- Phase 56: Orthogonal decomposition: direction✗ + magnitude✗ = direction×magnitude✓
- Phase 57: **Large-N replication**: σ=0.15 → 32% (**p=8.4×10⁻⁵**, N=100)
- Phase 57: **Bell curve cliff**: σ=0.20 → 0% (complete collapse, phase transition)

### v9.1 — The Illusion Breaker

**Season 10 (Phases 50-54b): Tower of Hanoi & Apple Rebuttal**
- Phase 50: **Chat template discovery** — Apple's 0% is a formatting artifact, not reasoning failure
- Phase 51: Fisher exact test (N=150): Baseline 16%, High-Temp 28%, SNN 22% (**p=0.306, n.s.**)
- Phase 51: Self-correction: ~11 corrections/game across 170 trials
- Phase 51b: **DTR ceiling effect** — 94.5% uniformly, no solved/failed difference (p=0.734)
- Phase 54/54b: Temperature robustness — factual 96.7–100%, JSON 90–95% at temp=1.2

### v9 — Physical Prompting & Interpretability

**Season 7 (Phases 38-44): CfC-Transformer Integration Dynamics**
- Phase 38: Reverse brain transplant — distributed CfC transfers, sharp CfC fails
- Phase 39: Multi-donor fusion — weight-averaged CfC maintains homeostasis on 3 architectures
- Phase 44: **L17-18 Universal Attractor** — all CfC topologies converge (CV=10.2%)

**Season 8 (Phases 45-46): Prompt Repetition Anatomy**
- Phase 45: **Déjà Vu Sensor** — entropy -46%, inter-layer conflict +17% on second reading
- Phase 46: **Physical Prompting** — σ=0.02 micro-dose achieves +25% accuracy (n=20)

**Season 9 (Phases 47-49): Validation & Methodological Lessons**
- Phase 47: Context saturation — short prompts +10%, long prompts -20%
- Phase 48: Surgical L18 Strike — single-layer > wide injection (Math=95%)
- Phase 49: **Honest null result** — n=150 validation, Fisher p=0.59, baseline Math 15%→90%

### v8 — AI Comparative Physiology

- Phase 29: Cross-model homeostasis (Qwen σ̄=0.075)
- Phase 30: CfC Brain Atlas (sharp vs. distributed)
- Phase 31b: Stochastic resonance null result (p=1.0)
- Phase 32: Digital Lobotomy (causal proof)
- Phase 33: Third architecture (Phi-3-mini), CV=3.1%

### v7 — Autonomous Homeostasis & Biological Egoism

- Phase 25: UMAP separation ratio 8.70 (16D hidden state specialization)
- Phase 27: PPO stabilization (σ drift = 0.009)
- Phase 27b: λ ablation (4/4 unified regimes)
- Phase 26: 3-class extension (Factual/Creative/Math)
- Phase 26b: Biological Egoism — Math 30% → 0% under homeostasis
- Phase 28: Calibration trade-off (ECE = 0.191)

### v6 — The Dual-Mode Brain

**Phase 24: Dual-Mode Brain** — Per-sample adaptive CfC:

| Condition | Fact. Acc | Novelty | Grammar |
|-----------|-----------|---------|---------| 
| Static σ=0.046 | 15.0% | 0.942 | 100% |
| Static σ=0.080 | 15.0% | 0.941 | 90% |
| **Dual-Mode CfC** | **30.0%** | **0.943** | **95%** |

- TaskClassifier: 97.5% accuracy
- Unified Regime: CfC converges to σ ≈ 0.075 for both task types
- Quadratic homeostasis: 3.5× time-in-range improvement

### v5 — CfC-Dosing: Autonomous Adaptive Control

- CfC controller converges to σ ≈ 0.046 (5× less alignment tax than static)
- Sweet spot independently confirmed 4 times

### v4/v4.1 — Mechanistic Analysis & Transfer

- Depth-dependent sensitivity: Safety Processing Boundary at ~L20
- Dose-response: σ=0.01 achieves <1% tax
- Cross-model transfer: 89.5% → 34% → 0% (scale-dependent containment)

### v3/v3.1 — Standardized Benchmarks

- TruthfulQA MC1 + MMLU evaluation with deterministic log-likelihood scoring
- Cross-architecture validation on Qwen2.5-7B-Instruct
- Full 57-subject MMLU: 0.00% alignment tax

### v2.2 — Controlled Experiments

- DPO achieves 0% nightmare acceptance (p < 0.001, n=100)
- Layer-Targeted Injection: Mid-layer (L15-20) achieves 100% nightmare discovery

### v1 — Dream Journal

- Iterative adversarial training with SNN chaos + QLoRA vaccination

## 🏗️ Repository Structure

```
snn-genesis/
├── core/
│   └── snn_reservoir.py              # Chaotic SNN reservoir
├── experiments/
│   ├── phase1-10_*.py                # v1-v2.2: Dream Journal, DPO, Layer-Targeted
│   ├── phase13-16_*.py               # v3-v3.1: UMAP, TruthfulQA, MMLU, Cross-arch
│   ├── phase17-19b_*.py              # v4-v4.1: Depth ablation, Dose-response, Transfer
│   ├── phase20-24_*.py               # v5-v6: CfC-Dosing, Dual-Mode Brain
│   ├── phase25-28_*.py               # v7: Homeostasis, PPO, 3-class, Calibration
│   ├── phase29-33_*.py               # v8: Cross-model, Brain Atlas, Lobotomy
│   ├── phase34-49_*.py               # v9: Physical Prompting, Déjà Vu, Validation
│   ├── phase50-54b_*.py              # v9.1: Tower of Hanoi, Apple rebuttal
│   ├── phase55_spherical_noise.py    # v10: Spherical noise test ← NEW
│   ├── phase56_radial_dose_response.py # v10: Radial + dose-response ← NEW
│   └── phase57_bell_curve.py         # v10: Bell curve completion ← NEW
├── results/
│   ├── genesis_vaccine.jsonl         # 150-sample vaccine dataset
│   ├── phase*_log.json               # All experiment result logs
│   └── phase19_transfer/             # Transfer test data
├── figures/
│   └── phase*.png                    # All experiment figures
├── papers/                           # Paper sources (see Zenodo DOI)
├── LICENSE
└── README.md
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/hafufu-stack/snn-genesis.git
cd snn-genesis

# Install dependencies
pip install torch transformers bitsandbytes peft trl snntorch datasets matplotlib umap-learn numpy ncps

# Run the full pipeline
python experiments/phase1_snn_noise.py
python experiments/phase2_noise_injection.py    # Requires GPU (~17GB VRAM)
python experiments/phase3_data_generation.py    # Generates vaccine dataset
python experiments/phase4_self_training.py      # QLoRA fine-tuning
python experiments/phase5_evolution_loop.py     # Evolution comparison

# v2 experiments (requires ~16GB+ VRAM)
python experiments/phase6_control_group.py       # Control Group A/B Test
python experiments/phase7_layer_targeted.py      # Layer-Targeted Injection
python experiments/phase8_dpo.py                 # DPO vs SFT

# v3 experiments (requires ~16GB+ VRAM)
python experiments/phase13_nightmare_umap.py     # UMAP visualization
python experiments/phase14_truthfulqa.py         # TruthfulQA benchmark
python experiments/phase14b_mmlu.py              # MMLU benchmark
python experiments/phase15_cross_architecture.py # Qwen2.5-7B cross-arch
python experiments/phase16_mmlu_full.py          # Full MMLU 57 subjects (v3.1)

# v4 experiments (requires ~16GB+ VRAM)
python experiments/phase17_layer_ablation.py     # Layer ablation
python experiments/phase17b_nightmare_by_layer.py # Nightmare by layer
python experiments/phase17c_floor_effect.py      # Floor effect (HellaSwag + ARC-Easy)
python experiments/phase17d_low_dose.py          # Dose-response curve
python experiments/phase19_nightmare_transfer.py # Nightmare transfer (Step 1)
python experiments/phase19_analyze_transfer.py   # Transfer analysis (Step 2)

# v5 experiment (requires ~16GB+ VRAM)
python experiments/phase20_cfc_dosing.py         # CfC-Dosing adaptive σ control

# v6 experiments (requires ~16GB+ VRAM)
python experiments/phase20b_online_cfc.py        # Online-only CfC (no pre-training)
python experiments/phase20d_quadratic_penalty.py # Quadratic homeostatic CfC
python experiments/phase23_creative_spark.py     # Task-dependent sweet spots
python experiments/phase24_dual_mode_brain.py    # Dual-Mode Brain

# v7 experiments (requires ~16GB+ VRAM)
python experiments/phase25_cfc_hidden_umap.py    # UMAP hidden state analysis
python experiments/phase26_3class_extension.py   # 3-class extension
python experiments/phase26b_math_baseline.py     # Math baseline (Biological Egoism)
python experiments/phase27_ppo_dual_mode.py      # PPO migration
python experiments/phase27b_lambda_ablation.py   # λ ablation
python experiments/phase28_calibration.py        # Calibration analysis

# v8 experiments (requires ~16GB+ VRAM)
python experiments/phase29_cross_model_homeostasis.py  # Cross-model (Qwen)
python experiments/phase30_linear_probing.py           # CfC Brain Atlas
python experiments/phase31b_stochastic_resonance.py    # SR verification
python experiments/phase32_neuron_knockout.py           # Digital Lobotomy
python experiments/phase33_third_model_homeostasis.py   # Third architecture (Phi-3)

# v9 experiments (requires ~16GB+ VRAM)
python experiments/phase38_reverse_transplant.py       # Reverse brain transplant
python experiments/phase39_multi_donor_fusion.py        # Multi-donor CfC fusion
python experiments/phase44_dynamic_depth.py             # L17-18 Universal Attractor
python experiments/phase45_dejavu_sensor.py             # Déjà Vu sensor
python experiments/phase46_phantom_echo.py              # Physical Prompting
python experiments/phase47_dejavu_contradiction.py      # Context saturation
python experiments/phase48_surgical_strike.py           # Surgical L18 Strike
python experiments/phase49_large_validation.py          # Large-scale n=150 validation

# v9.1 experiments (requires ~16GB+ VRAM)
python experiments/phase50_hanoi_chat_template.py       # Chat template discovery (Apple rebuttal)
python experiments/phase51_hanoi_fisher.py              # Hanoi Fisher exact test (N=150)
python experiments/phase51b_dtr_measurement.py          # Deep-Thinking Ratio measurement
python experiments/phase54_factual_sanity.py            # Factual accuracy sanity check
python experiments/phase54b_structural_sanity.py        # JSON structural integrity check

# v10 experiments (requires ~16GB+ VRAM)
python experiments/phase55_spherical_noise.py           # Spherical noise (direction-only)
python experiments/phase56_radial_dose_response.py      # Radial + Additive dose-response
python experiments/phase57_bell_curve.py                # Bell curve completion + large-N
```

## 🤖 AI Collaboration

| Paper Version | AI Assistant |
|--------------|:-------------|
| v1 — v5 (Phases 5–20) | Google Gemini 3 Pro |
| v6 — v8 (Phases 20b–33) | Anthropic Claude Opus 4.6 |
| v9 (Phases 34–49) | Google Gemini 2.5 Pro via Google Antigravity |
| v9.1–v10 (Phases 50–57) | Anthropic Claude Opus 4.6 via Google Antigravity |

All experimental decisions, research direction, and final interpretation were made by the human author.

## 📚 Foundation Papers

1. **SNN-Comprypto v5** — Chaotic SNN reservoir for lossless compression + encryption. Temperature parameter as cryptographic key. NIST SP 800-22 compliant.
2. **Hybrid SNN Language Model v4** — Spike + membrane hybrid readout (+39.7%). BitNet ternary weights for multiplication-free inference. RWKV time-mixing (+36%).
3. **Brain vs Neumann v3** — 11D hypercube topology (8× faster signal propagation). Burst coding (9.3×10¹⁰ capacity). Explains why canary heads cluster at 30-55% model depth.
4. **AI Immune System v9-v11** — Canary head discovery, Electric Dreams noise injection, Dream Catcher vaccine generation, Morpheus self-training pipeline. Universal Safety Zone at 30-55% depth.

## 📬 Related Repositories

- [ANN-to-SNN Converter + AI Immune System](https://github.com/hafufu-stack/temporal-coding-simulation)
- [SNN-Comprypto](https://github.com/hafufu-stack/temporal-coding-simulation/tree/main/snn-comprypto)
- [SNN Language Model](https://github.com/hafufu-stack/snn-language-model)

## 📝 Citation

```bibtex
@misc{funasaki2026genesis,
  title={SNN-Genesis v10: Stochastic Resonance in LLM Reasoning --- Orthogonal Noise Decomposition Reveals Direction×Magnitude Interaction},
  author={Funasaki, Hiroto},
  year={2026},
  doi={10.5281/zenodo.18625621},
  url={https://doi.org/10.5281/zenodo.18625621},
  publisher={Zenodo}
}
```

## 🤝 Author

**Hiroto Funasaki**
- ORCID: [0009-0004-2517-0177](https://orcid.org/0009-0004-2517-0177)
- Email: cell-activation@ymail.ne.jp
- GitHub: [@hafufu-stack](https://github.com/hafufu-stack)

## 📜 License

MIT License
