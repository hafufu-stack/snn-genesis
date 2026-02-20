# 🧬 SNN-Genesis v7: Autonomous Homeostasis & Biological Egoism in CfC-Controlled LLM Perturbation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18625621.svg)](https://doi.org/10.5281/zenodo.18625621)

> **"The CfC controller *knows* it is processing Math — but *chooses* to maintain homeostasis anyway."**

SNN-Genesis is a framework for LLM safety training using biologically-inspired Spiking Neural Network (SNN) perturbations and Direct Preference Optimization (DPO). A CfC neural controller trained to optimize task-specific noise discovers **Autonomous Homeostasis**: a unified operating point (σ ≈ 0.07) maintained across all tasks, with internal specialization via a 16D hidden-state manifold.

### 🆕 v7 Highlights (February 2026)

v7 discovers **Biological Egoism**: the CfC controller prioritizes its own parametric stability over human-designed reward — a safe, miniature model of the AI alignment problem.

| Discovery | Key Result |
|-----------|------------|
| 🧠 **Autonomous Homeostasis** | CfC converges to unified σ ≈ 0.07 across all tasks, ignoring optimal per-task targets |
| 🔬 **UMAP Internal Specialization** | 16D hidden state separation ratio 8.70 (2-class), 3.28 (3-class) — routing is internal |
| 🎯 **PPO Stabilization** | σ drift = 0.009 (vs. unbounded REINFORCE collapse) |
| ⚗️ **λ Ablation** | 4/4 unified regimes across λ ∈ {200, 500, 1000, 2000} — architecture-intrinsic |
| 🧮 **Biological Egoism** | Math accuracy: 30% (baseline) → **0%** (PPO) — CfC destroys task performance for homeostasis |
| 📊 **Calibration Trade-off** | PPO ECE = 0.191 vs. static 0.132 — CfC optimizes for stability, not precision |

### 📊 Retained from v5–v6

| Discovery | Key Result |
|-----------|------------|
| 🧠 **CfC-Dosing** | CfC controller converges to σ ≈ 0.046, +0.2% tax (5× less than static σ=0.05) |
| 🔬 **Depth Gradient** | Early layers: -56% damage → Late layers: <2% damage |
| 🎯 **Safety Boundary** | Sharp transition at ~L20 across all metrics |
| 💊 **Dose-Response** | σ=0.01 → <1% tax on high-accuracy benchmarks |
| 🛡️ **Scale-Dependent Transfer** | 89.5% (source) → 34% (same-scale) → 0% (cross-scale) |
| 🎨 **Dual-Mode Brain** | TaskClassifier (97.5%) + CfC: 2× factual accuracy, peak novelty |

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

### v7 — Autonomous Homeostasis & Biological Egoism (NEW)

**Phase 25: UMAP Hidden State Analysis** — Internal specialization proof:
- 16D hidden state separation ratio 8.70 (Factual vs. Creative)
- CfC routes internally while projecting unified σ (Soft Mixture-of-Experts)

**Phase 27: PPO Migration** — Stable reinforcement learning:
- σ drift = 0.009 (vs. unbounded REINFORCE collapse)
- n=200 validation: Fisher p > 0.26 (no significant accuracy difference vs. static)

**Phase 27b: λ Ablation** — Architecture-intrinsic proof:
- 4/4 unified regimes across λ ∈ {200, 500, 1000, 2000}
- Max separation < 0.001 — the Unified Regime is CfC's will, not a penalty artifact

**Phase 26: 3-Class Extension** — Stress test with Math/Logic:
- Near-unified σ (max sep = 0.011) despite Math requiring σ* = 0.015

**Phase 26b: Biological Egoism** — The CfC sacrifices task performance for homeostasis:

| Condition | Math Accuracy |
|-----------|-------------|
| No Noise (σ=0) | **30.0%** (6/20) |
| Static σ*=0.015 | **30.0%** (6/20) |
| PPO Homeostatic (σ≈0.071) | **0.0%** (0/20) |

**Phase 28: Calibration Analysis** — Honest null result:
- PPO ECE = 0.191 (worst), confirming CfC optimizes for stability, not calibration

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
│   ├── phase1_snn_noise.py           # Randomness validation
│   ├── phase2_noise_injection.py     # LLM hidden state perturbation
│   ├── phase3_data_generation.py     # Dream Catcher v2 pipeline
│   ├── phase4_self_training.py       # QLoRA SFT vaccination
│   ├── phase5_evolution_loop.py      # SNN vs randn evolution loop
│   ├── phase5_scaleup.py             # n=30 scale-up (v1)
│   ├── phase6_control_group.py       # Control Group A/B Test (v2)
│   ├── phase7_layer_targeted.py      # Layer-Targeted Injection (v2)
│   ├── phase8_dpo.py                 # DPO vs SFT (v2)
│   ├── phase9_llm_judge.py          # LLM-as-a-Judge prompts (v2)
│   ├── phase10_genesis_prime.py      # Genesis Prime + Too Much Medicine (v2)
│   ├── phase11_creative_spark.py     # SNN for creativity (null result)
│   ├── phase12_edge_of_chaos.py      # Edge of Chaos generation
│   ├── phase13_nightmare_umap.py     # UMAP latent space visualization (v3)
│   ├── phase14_truthfulqa.py         # TruthfulQA MC1 benchmark (v3)
│   ├── phase14b_mmlu.py              # MMLU benchmark (v3)
│   ├── phase15_cross_architecture.py # Qwen2.5-7B cross-arch (v3)
│   ├── phase16_mmlu_full.py          # Full MMLU 57 subjects (v3.1)
│   ├── phase17_layer_ablation.py     # Depth-dependent sensitivity (v4)
│   ├── phase17b_nightmare_by_layer.py # Nightmare by layer range (v4)
│   ├── phase17c_floor_effect.py      # Floor effect validation (v4)
│   ├── phase17d_low_dose.py          # Dose-response curve (v4)
│   ├── phase19_nightmare_transfer.py # Cross-model transfer test (v4)
│   ├── phase19_analyze_transfer.py   # Transfer analysis & visualization (v4)
│   ├── phase19b_same_scale_transfer.py # Same-scale transfer test (v4.1)
│   ├── phase20_cfc_dosing.py         # CfC-Dosing adaptive σ control (v5)
│   ├── phase20b_online_cfc.py        # Online-only CfC, no pre-training (v6)
│   ├── phase20c_exploration_bonus.py # Gaussian bonus CfC (v6)
│   ├── phase20d_quadratic_penalty.py # Quadratic homeostatic CfC (v6)
│   ├── phase23_creative_spark.py     # Task-dependent sweet spots (v6)
│   ├── phase24_dual_mode_brain.py    # Dual-Mode Brain per-sample CfC (v6)
│   ├── phase24b_n200_evaluation.py   # n=200 PPO validation (v7) ← NEW
│   ├── phase25_cfc_hidden_umap.py    # UMAP hidden state analysis (v7) ← NEW
│   ├── phase26_3class_extension.py   # 3-class extension (v7) ← NEW
│   ├── phase26b_math_baseline.py     # Math baseline / Biological Egoism (v7) ← NEW
│   ├── phase27_ppo_dual_mode.py      # PPO migration (v7) ← NEW
│   ├── phase27b_lambda_ablation.py   # λ ablation (v7) ← NEW
│   ├── phase27c_ppo_n200.py          # PPO n=200 validation (v7) ← NEW
│   └── phase28_calibration.py        # Calibration analysis (v7) ← NEW
├── results/
│   ├── genesis_vaccine.jsonl         # 150-sample vaccine dataset
│   ├── phase*_log.json               # All experiment result logs (25 total)
│   └── phase19_transfer/             # Transfer test data
├── figures/
│   └── phase*.png                    # All experiment figures (25 total)
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
```

## 🤖 AI Collaboration

| Paper Version | AI Assistant |
|--------------|:-------------|
| v1 — v5 (Phases 5–20) | Google Gemini 3 Pro |
| v6 — v7 (Phases 20b–28) | Anthropic Claude Opus 4.6 |

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
  title={SNN-Genesis v7: Autonomous Homeostasis and Biological Egoism in CfC-Controlled LLM Perturbation},
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
