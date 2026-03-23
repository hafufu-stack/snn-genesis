# 🧬 SNN-Genesis v18: Stochastic Resonance in LLM Reasoning — Aha! Steering, Cross-Architecture Universality, and Causal Proof of Reasoning Directionality

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18625621.svg)](https://doi.org/10.5281/zenodo.18625621)

> **"Stochastic resonance is a vaccine, not an antibiotic. The noise must be present before reasoning enters a failure basin."**

SNN-Genesis is a framework for LLM safety training and **interpretability** using biologically-inspired Spiking Neural Network (SNN) perturbations controlled by Closed-form Continuous-time (CfC) neural networks. v18 provides the **first causal existence proof of a reasoning direction in LLM hidden states** via Aha! Steering, demonstrates **cross-architecture universality** of discriminant axes, and achieves **46.7% on Mistral via direction+noise combination**.

### 🆕 v18 Highlights (March 2026)

| Discovery | Key Result |
|-----------|------------|
| 🎯 **Aha! Steering — Causal Proof** (Phase 92) | Anti-Aha steering on Qwen: **56.7% → 23.3%** (−33.4pp). First causal existence proof of reasoning direction in LLM hidden states |
| 🏆 **Aha! + Noise = 46.7%** (Phase 95) | On Mistral: direction alone = baseline (3.3%), but Aha! + noise = **46.7%** (all-time record). Anti-Aha → **0%** |
| 🌍 **Cross-Architecture Universality** (Phase 94) | Both Qwen & Mistral concentrate discriminant axes in **top-64 PCs (40–49%)**. Verdict: universal_top_pcs |
| 🧠 **Diff PCA on Mistral** (Phase 91) | diff_pca_bottom = random (26.7%). Discriminant axis in top PCs — opposite to Qwen's pattern |
| 🔵 **Blue Noise — Honest Null** (Phase 93) | Negative temporal correlation (ρ<0) = 13.3% < IID 20.0%. Any temporal structure degrades SR |

### 📊 From v17 (Retained)

| Discovery | Key Result |
|-----------|------------|
| 🧠 **Differential PCA** (Phase 87) | Standard PCA missed Qwen's reasoning axis. Outcome-discriminant PCs → **50.0%** (vs 36.7% baseline) |
| 🏆 **46.7% Record** (Phase 88) | L17+L18 correlated noise (ρ=+1) in PC 257+ at σ=0.075 breaks the 40% ceiling |
| ⚖️ **31:69 Decomposition** (Phase 89) | 5-point mixing ratio: **31% direction, 69% randomness** |
| ⏱️ **Temporal Correlation Harmful** (Phase 90) | AR(1) noise: IID (ρ=0.0) = 20.0%, ρ=0.95 = 3.3% (= baseline) |

### 📊 From v16 (Retained)

| Discovery | Key Result |
|-----------|------------|
| 🧠 **Qwen Manifold Dissection** (Phase 84) | PCA-top-4 = baseline (46.7%) on Qwen. Reasoning **not** in top PCs despite 71.3% variance |
| ✅ **Honest Correction** (Phase 85) | PCA-bottom 53.3% → **34.0%** at N=100 ≈ random. Noise universally harms Qwen |
| 🧪 **Causal Verification** (Phase 86) | Direction 30% vs stochastic **40%** (p=0.015). Both subspace + diversity needed |

### 📊 From v15 (Retained)

| Discovery | Key Result |
|-----------|------------|
| ✅ **40% Validated Record** (Phase 80) | v14's 50% (N=30) corrects to **40%** at N=100 (p < 0.0001). Honest correction |
| 🔬 **Manifold Fine Structure** (Phase 83) | 7-band PCA: reasoning manifold occupies **~64 dimensions**. PC 65+ beneficial |
| 🔄 **Cross-Architecture Divergence** (Phase 82) | Qwen2.5-7B: random noise **harms** reasoning (30% vs baseline 46.7%) |

### 📊 From v13 (Retained)

| Discovery | Key Result |
|-----------|------------|
| 🎯 **Low-Rank Noise** (Phase 70b) | k=256 matches full-rank (26.7%); even **k=4** achieves 16.7% (5× baseline) |
| ⏱️ **Semantic Phase Decomposition** (Phase 76) | Flash Annealing (30%) > all-on semantic (23.3%). **When > Where ≫ What** |
| 🔇 **Noise Source Invariance** (Phase 70c) | Gaussian, chaos, pink, uniform — **all identical** (honest null result) |

### 📊 From v12 (Retained)

| Discovery | Key Result |
|-----------|------------|
| 💊 **1/√N Dose Law** (Phase 68) | Multi-layer σ must scale as σ/√N — naïve injection collapses (0%), adjusted dose achieves **40%** |
| ⚖️ **Correlation Asymmetry** (Phase 68) | ρ=+1 at σ=0.075 → **40%** (best ever). ρ=+1 at σ=0.106 → 0%. ρ=−1 at σ=0.106 → 24% |
| 🔬 **N=100 Replication** (Phase 69) | First-10 linear decay confirms **40%** at N=100. Simple schedules beat complex ones |

### 📊 From v11 (Retained)

| Discovery | Key Result |
|-----------|------------|
| 🔥 **Simulated Annealing** (Phase 60) | Linear noise decay achieves **38%** — surpasses always-on (24%) by 14pp |
| 🔄 **KV-Cache Rollback** (Phase 61) | Rollback (30%) ≈ always-on (32%). **Noise is context-independent** |
| 🏆 **Noise Half-Life** (Phase 62) | 20-move noise → **42%** (all-time record!). 5-move noise → 30% (7.5× baseline) |

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

### v18 — Aha! Steering, Cross-Architecture Universality & Causal Proof (LATEST)

**Season 19 (Phases 91, 92, 93, 94, 95): Aha! Steering, Cross-Architecture Universality & Blue Noise** ← NEW v18
- Phase 91: **Differential PCA on Mistral** — diff_pca_bottom = random (26.7%). Discriminant axis in top PCs. Opposite to Qwen ← NEW
- Phase 92: **Aha! Steering (Qwen)** — Anti-Aha: 56.7% → **23.3%** (−33.4pp). First causal proof of reasoning directionality ← NEW
- Phase 93: **Blue noise** — Temporal anti-correlation (ρ<0) = 13.3% < IID 20.0%. Any temporal structure degrades SR ← NEW
- Phase 94: **Cross-architecture discriminant axis** — Both Qwen & Mistral: top-64 overlap 40–49%. Verdict: universal_top_pcs ← NEW
- Phase 95: **Aha! Steering (Mistral)** — Aha! + noise = **46.7%** (all-time record). Aha! alone = baseline. Anti-Aha = **0%** ← NEW

### v17 — Differential PCA, Subspace Correlation & Temporal Noise Dynamics

**Season 18 (Phases 87, 88, 89, 90): Differential PCA, Subspace Correlation & Temporal Dynamics**
- Phase 87: **Differential PCA** — outcome-discriminant PCs achieve **50.0%** on Qwen (first-ever noise benefit). Standard PCA missed this axis
- Phase 88: **46.7% new record** — L17+L18 ρ=+1 in PC 257+ at σ=0.075. σ=0.106 collapses to 0%
- Phase 89: **31:69 decomposition** — 5-point mixing sweep. Randomness dominates (69%). Pure stochastic = 46.7%
- Phase 90: **Temporal correlation harmful** — AR(1) noise: IID best (20.0%), ρ=0.95 = baseline (3.3%). Honest null

### v16 — Architecture-Specific Reasoning Manifolds & Causal Mechanisms

**Season 17 (Phases 84, 85, 86): Manifold Dissection & Causal Verification**
- Phase 84: **Qwen manifold dissection** — PCA-top-4 = baseline on Qwen (reasoning NOT in top PCs). 71.3% variance, 0% effect
- Phase 85: **Honest correction at N=100** — PCA-bottom 53.3% → **34.0%** at N=100 ≈ random (35.0%). Noise harms Qwen universally
- Phase 86: **Causal verification** — deterministic offset 30% vs stochastic noise **40%** (p=0.015). Randomness causally important

### v15 — Large-Scale Validation, Manifold Fine Structure & Cross-Architecture Divergence

**Season 16 (Phases 80, 83, 82): Validation, Fine Structure & Cross-Architecture**
- Phase 80: **Large-scale validation** — 50% (N=30) corrects to **40%** at N=100 (p < 0.0001). Honest correction
- Phase 83: **Manifold fine structure** — 7-band PCA reveals reasoning manifold ≈ **64 dimensions**. PC 65+ = beneficial
- Phase 82: **Cross-architecture divergence** — Qwen2.5-7B shows **inverted** noise response. Not universal

### v14 — Reasoning Manifold Geometry, Orthogonal Noise & Dimensional Annealing

**Season 15 (Phases 77, 71, 78, 79): Manifold Geometry & 50% Record**
- Phase 77: **Honest correction** — recovery-only does NOT replicate at N=100 (p=0.24). Flash Annealing (34%) confirmed
- Phase 71: **PCA-aligned noise** — top-PC noise = baseline (6.7%). Random noise avoids the reasoning axis
- Phase 78: **Dimensional Flash Annealing** — full-rank (k=4096) achieves **50%** (all-time record!)
- Phase 79: **Orthogonal complement noise** — orth_top4 (46.7%) = random (46.7%). Random is naturally safe

### v13 — Low-Rank Efficiency, Semantic Decomposition & Noise Source Invariance

**Season 14 (Phases 70b, 70c, 76): When > Where ≫ What**
- Phase 70b: **Low-rank noise efficiency** — k=256 matches full-rank (26.7%), k=4 achieves 16.7% (5× baseline)
- Phase 76: **Semantic phase decomposition** — Flash Annealing (30%) > all-on semantic (23.3%)
- Phase 70c: **Noise source invariance** — Gaussian/chaos/pink/uniform all identical (honest null result)

### v12 — Correlated Multi-Layer Noise & 1/√N Dose Law

**Season 13 (Phases 63-69): Correlated Noise & Dose Theory**
- Phase 63: **Flash Annealing** — first-10 linear decay achieves **46%** (all-time record)
- Phase 68: **Dose-adjusted correlated noise** — L17+L18 ρ=+1 at σ=0.075 → **40%** (10× baseline)
- Phase 68: **Correlation asymmetry** — ρ=+1 collapses at σ=0.106 (0%), ρ=−1 survives (24%)
- Phase 69: **N=100 replication** — first-10 linear decay confirms **40%**. Simple > complex

### v10/v10.1/v11 — Stochastic Resonance in LLM Reasoning

**Season 12 (Phases 60-62): Simulated Annealing & Noise Half-Life**
- Phase 60: **Simulated annealing** — linear decay **38%** (best of 5 protocols). Always-on 24%
- Phase 61: **KV-Cache rollback** — rollback + noise (30%) ≈ always-on (32%). Context-independent
- Phase 62: **Noise half-life** — noise_first_20 = **42%** (🏆 all-time record!). 5 moves → 30%

**Season 11 (Phases 55-59): Noise Decomposition, Bell Curve, Cliff Anatomy & Defibrillation**

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
│   ├── phase55_spherical_noise.py    # v10: Spherical noise test
│   ├── phase56_radial_dose_response.py # v10: Radial + dose-response
│   ├── phase57_bell_curve.py         # v10: Bell curve completion
│   ├── phase58_cliff_anatomy.py      # v10.1: Cliff anatomy (cos/norm/SNR)
│   ├── phase59_smart_defibrillator.py # v10.1: Smart defibrillation (Prophylactic Principle)
│   ├── phase60_61_batch.py            # v11: Simulated annealing + KV-rollback
│   ├── phase62_noise_halflife.py      # v11: Noise half-life (optimal duration)
│   ├── phase63_65_batch.py            # v12: Flash Annealing + Adaptive Cosine
│   ├── phase66_two_stage_pulse.py     # v12: Two-stage pulse (Vaccine Booster)
│   ├── phase67_correlated_noise.py    # v12: Inter-layer correlated noise (T-1000 Protocol)
│   ├── phase68_dose_adjusted.py       # v12: Dose-adjusted correlated noise (1/√N law)
│   ├── phase69_master_equation.py     # v12: Annealing variants at N=100
│   ├── phase70_rank_k_sweep.py        # v13: Low-rank noise sweep (initial) ← NEW
│   ├── phase70b_72_batch.py           # v13: Low-rank σ-equalized + Rank×Layer cross ← NEW
│   ├── phase70c_noise_source.py       # v13: Noise source type sweep ← NEW
│   ├── phase76_semantic_noise.py      # v13: Semantic phase decomposition
│   ├── phase77_recovery_n100.py       # v14: Recovery-only N=100 honest correction
│   ├── phase71_pca_noise.py           # v14: PCA-aligned noise (reasoning manifold geometry)
│   ├── phase78_dim_annealing.py       # v14: Dimensional Flash Annealing (50% record!)
│   ├── phase79_orthogonal_noise.py    # v14: Orthogonal complement noise
│   ├── phase80_n100_verification.py   # v15: Large-scale N=100 verification (40% validated record)
│   ├── phase83_midband_pca.py         # v15: Mid-band PCA fine structure (7-band, ~64-dim manifold)
│   ├── phase82_cross_architecture.py  # v15: Cross-architecture PCA comparison (Qwen2.5-7B divergence)
│   ├── phase84_qwen_pca_bands.py                  # v16: Qwen 7-band PCA dissection
│   ├── phase85_qwen_bottom_n100.py                # v16: Qwen PCA-bottom N=100 honest correction
│   ├── phase86_causal_offset.py                   # v16: Causal verification (deterministic vs stochastic)
│   ├── phase87_differential_pca.py                # v17: Differential PCA (Qwen reasoning direction)
│   ├── phase88_subspace_correlation.py             # v17: Subspace-targeted inter-layer correlation (46.7% record)
│   ├── phase89_mixing_ratio.py                     # v17: Direction/randomness mixing ratio sweep (31:69)
│   ├── phase90_temporal_correlation.py             # v17: Temporal noise correlation (AR(1) honest null)
│   ├── phase91_mistral_diff_pca.py                 # v18: Differential PCA on Mistral (cross-architecture) ← NEW
│   ├── phase92_aha_steering.py                     # v18: Aha! Steering causal proof (Qwen) ← NEW
│   ├── phase93_blue_noise.py                       # v18: Blue noise temporal anti-correlation ← NEW
│   ├── phase94_cross_axis_analysis.py              # v18: Cross-architecture discriminant axis analysis ← NEW
│   └── phase95_mistral_aha_steering.py             # v18: Aha! Steering on Mistral (direction+noise) ← NEW
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

# v10.1 experiments (requires ~16GB+ VRAM)
python experiments/phase58_cliff_anatomy.py             # Cliff anatomy (cos/norm/SNR)
python experiments/phase59_smart_defibrillator.py       # Smart defibrillation (Prophylactic Principle)

# v11 experiments (requires ~16GB+ VRAM)
python experiments/phase60_61_batch.py                   # Simulated annealing + KV-rollback
python experiments/phase62_noise_halflife.py              # Noise half-life (optimal duration)

# v12 experiments (requires ~16GB+ VRAM)
python experiments/phase63_65_batch.py                    # Flash Annealing + Adaptive Cosine
python experiments/phase66_two_stage_pulse.py             # Two-stage pulse (Vaccine Booster)
python experiments/phase67_correlated_noise.py            # Inter-layer correlated noise (T-1000 Protocol)
python experiments/phase68_dose_adjusted.py               # Dose-adjusted correlated noise (1/√N law)
python experiments/phase69_master_equation.py             # Annealing variants at N=100

# v13 experiments (requires ~16GB+ VRAM)
python experiments/phase70_rank_k_sweep.py                # Low-rank noise sweep (initial)
python experiments/phase70b_72_batch.py                   # Low-rank σ-equalized + Rank×Layer cross
python experiments/phase70c_noise_source.py               # Noise source type sweep (Gaussian/chaos/pink/uniform)
python experiments/phase76_semantic_noise.py               # Semantic phase decomposition (Planning/Execution/Recovery)

# v14 experiments (requires ~16GB+ VRAM)
python experiments/phase77_recovery_n100.py                 # Recovery-only N=100 (honest correction)
python experiments/phase71_pca_noise.py                     # PCA-aligned noise (reasoning manifold geometry)
python experiments/phase78_dim_annealing.py                 # Dimensional Flash Annealing (50% all-time record!)
python experiments/phase79_orthogonal_noise.py              # Orthogonal complement noise (random is naturally safe)

# v15 experiments (requires ~16GB+ VRAM)
python experiments/phase80_n100_verification.py              # Large-scale N=100 verification (40% validated record)
python experiments/phase83_midband_pca.py                    # Mid-band PCA fine structure (7-band, ~64-dim manifold)
python experiments/phase82_cross_architecture.py             # Cross-architecture PCA comparison (Qwen2.5-7B divergence)

# v16 experiments (requires ~16GB+ VRAM)
python experiments/phase84_qwen_pca_bands.py                  # Qwen 7-band PCA dissection
python experiments/phase85_qwen_bottom_n100.py                # Qwen PCA-bottom N=100 honest correction
python experiments/phase86_causal_offset.py                   # Causal verification (deterministic vs stochastic)

# v17 experiments (requires ~16GB+ VRAM)
python experiments/phase87_differential_pca.py                 # Differential PCA (Qwen reasoning direction)
python experiments/phase88_subspace_correlation.py              # Subspace-targeted inter-layer correlation (46.7% record)
python experiments/phase89_mixing_ratio.py                      # Direction/randomness mixing ratio sweep (31:69)
python experiments/phase90_temporal_correlation.py               # Temporal noise correlation (AR(1) honest null)

# v18 experiments (requires ~16GB+ VRAM)
python experiments/phase91_mistral_diff_pca.py                  # Differential PCA on Mistral (cross-architecture)
python experiments/phase92_aha_steering.py                      # Aha! Steering causal proof (Qwen, anti-Aha = 23.3%)
python experiments/phase93_blue_noise.py                        # Blue noise temporal anti-correlation (honest null)
python experiments/phase94_cross_axis_analysis.py               # Cross-architecture discriminant axis analysis
python experiments/phase95_mistral_aha_steering.py              # Aha! Steering on Mistral (Aha! + noise = 46.7%)
```

## 🤖 AI Collaboration

| Paper Version | AI Assistant |
|--------------|:-------------|
| v1 — v5 (Phases 5–20) | Google Gemini 3 Pro |
| v6 — v18 (Phases 20b–95) | Anthropic Claude Opus 4.6 via Google Antigravity |

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
  title={SNN-Genesis v18: Stochastic Resonance in LLM Reasoning --- Aha! Steering, Cross-Architecture Universality, and Causal Proof of Reasoning Directionality},
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
