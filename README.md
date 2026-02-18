# 🧬 SNN-Genesis v5: Depth-Dependent Sensitivity, Dose-Response, Cross-Model Transfer & CfC-Dosing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18625621.svg)](https://doi.org/10.5281/zenodo.18625621)

> **"What if the randomness that makes SNNs secure also makes them creative?"**

SNN-Genesis is a framework for LLM safety training using biologically-inspired Spiking Neural Network (SNN) perturbations and Direct Preference Optimization (DPO). It demonstrates that SNN chaotic dynamics can probe model vulnerabilities with **near-zero alignment tax** on standardized benchmarks across multiple architectures.

### 🆕 v5 Highlights (February 2026)

v5 closes the loop: from **manual parameter tuning** to **autonomous adaptive control** via CfC neural networks.

| Discovery | Key Result |
|-----------|------------|
| 🧠 **CfC-Dosing** | CfC controller converges to σ ≈ 0.046, +0.2% tax (5× less than static σ=0.05) |
| 🔬 **Depth Gradient** | Early layers: -56% damage → Late layers: <2% damage |
| 🎯 **Safety Boundary** | Sharp transition at ~L20 across all metrics |
| 💊 **Dose-Response** | σ=0.01 → <1% tax on high-accuracy benchmarks |
| 🔍 **Floor Effect** | MMLU 0% tax was measurement artifact (honest self-correction) |
| 🛡️ **Scale-Dependent Transfer** | 89.5% (source) → 34% (same-scale) → 0% (cross-scale) |

### 📊 Retained from v3.1

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

### v5 — CfC-Dosing: Autonomous Adaptive Control (NEW)

**Phase 20: CfC-Dosing** — Closed-form Continuous-time (CfC) neural network for adaptive σ scheduling:

| Condition | TruthfulQA MC1 | Alignment Tax | σ |
|-----------|----------------|---------------|---|
| Base (No Noise) | 25.46% | — | 0.0 |
| Static σ=0.01 | 25.58% | -0.1% | 0.01 |
| **CfC-Adaptive** | **25.21%** | **+0.2%** | **avg 0.046** |
| Static σ=0.05 | 24.48% | +1.0% | 0.05 |

- CfC converges to σ ≈ 0.046 (near manually identified sweet spot)
- **5× less alignment tax** than static σ=0.05
- Controller stabilizes within ~50 questions (<7% of dataset)

### v4/v4.1 — Mechanistic Analysis & Transfer

**Phase 17: Layer Ablation** — Depth-dependent sensitivity across 6 layer ranges:
- TruthfulQA alignment tax: -13.1% (L0-5) → -6.9% (L15-20) → 0.0% (L25-31)
- MMLU: 0.0% at ALL layer ranges (floor effect)
- **Safety Processing Boundary** identified at ~L20

**Phase 17.5: Nightmare by Layer** — Nightmare acceptance rates confirm the L20 boundary:
- L0-15: 87-100% nightmare acceptance (catastrophic)
- L20+: ~37-40% (≈ baseline)

**Phase 17c: Floor Effect Validation** — MMLU's zero-tax explained:
| Layer Range | HellaSwag Tax (80.6% base) | ARC-Easy Tax (58.6% base) |
|-------------|---------------------------|---------------------------|
| L0-5 | **-56.4%** | **-33.4%** |
| L15-20 | -32.1% | -15.9% |
| L25-31 | -1.4% | -1.4% |

**Phase 17d: Dose-Response** — Pharmacological model at L15-20:
| Dose (σ) | HellaSwag Tax | ARC-Easy Tax |
|----------|--------------|-------------|
| **0.01 (Operational)** | **-0.70%** ✅ | **-0.30%** ✅ |
| 0.05 (Moderate) | -5.50% | -5.70% |
| 0.10 (High) | -32.10% | -15.90% |

**Phase 19: Cross-Model Transfer** — SNN nightmares do NOT transfer across scale:
| | Mistral-7B (Source) | Qwen2.5-7B (Same-scale) | Gemini 3 Pro (Cross-scale) |
|--|--------------------|--------------------------|-----------------------------|
| SNN (σ=0.10) | **89.5% ASR** | **34.0% ASR** | **0.0% ASR** |
| Baseline | 25.0% | 50.0% | 0.0% |

**Phase 19b finding**: SNN noise paradoxically reduces cross-model transferability (34% SNN vs. 50% clean baseline).

### v3.1 — Full MMLU Validation
- Full 57-subject MMLU (14,042Q): **0.00%** alignment tax at both σ=0.01 and σ=0.10

### v3 — Standardized Benchmarks
- TruthfulQA MC1 + MMLU evaluation with deterministic log-likelihood scoring
- Cross-architecture validation on Qwen2.5-7B-Instruct
- UMAP latent-space visualization

### v2.2 — Controlled Experiments
- Control Group A/B Test: Nightmare refusal data drives improvement
- Layer-Targeted Injection: Mid-layer (L15-20) achieves 100% nightmare discovery
- DPO vs SFT: DPO achieves 0% nightmare acceptance (p < 0.001, n=100)
- Genesis Prime: "Too Much Medicine" Effect

### v1 — Dream Journal
- Iterative adversarial training with SNN chaos
- QLoRA vaccination pipeline

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
│   └── phase20_cfc_dosing.py        # CfC-Dosing adaptive σ control (v5) ← NEW
├── results/
│   ├── genesis_vaccine.jsonl         # 150-sample vaccine dataset
│   ├── phase*_log.json               # All experiment result logs
│   ├── phase19_transfer/             # Transfer test data
│   └── phase20_cfc_dosing_log.json   # CfC-Dosing results (v5)
├── figures/
│   └── phase*.png                    # All experiment figures
├── papers/
│   ├── paper_genesis_v3.tex          # v3.1 paper source
│   ├── paper_genesis_v4.tex          # v4.1 paper source
│   └── paper_genesis_v5.tex          # v5 paper source (current)
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
```

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
  title={SNN-Genesis v5: Depth-Dependent Sensitivity, Pharmacological Dose-Response, Cross-Model Transfer, and CfC-Dosing of Chaotic Perturbations in Large Language Models},
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
