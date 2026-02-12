# Project Genesis: Self-Evolving AI via SNN Randomness

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 🧬 **"What if the randomness that makes SNNs secure also makes them creative?"**
>
> SNN chaotic dynamics → high-quality noise → data augmentation → self-learning loop

## 💡 Core Idea

Spiking Neural Networks produce **cryptographically random** noise (0.388% prediction rate, matching theoretical random). This is 17-56× more random than ANNs/LSTMs.

**Project Genesis** exploits this for a self-evolving data pipeline:

```
Phase 1: SNN Random Noise Generation (from SNN-Comprypto)
    ↓
Phase 2: Inject noise into LLM hidden states (from Electric Dreams, v10)
    ↓
Phase 3: Generate nightmare/healed training pairs (from Dream Catcher, v10)
    ↓
Phase 4: Self-train via SFT (from Project Morpheus, v11)
    ↓
Phase 5: Better model generates better data → Positive feedback loop!
```

## 🔗 Building on 5 Papers

| Source Paper | Technology Used | Role in Genesis |
|-------------|----------------|-----------------|
| SNN-Comprypto v5 | Chaotic SNN randomness | High-quality noise source |
| AI Safety v10 | Electric Dreams | Noise injection framework |
| AI Safety v10 | Dream Catcher | Data generation pipeline |
| AI Safety v11 | Project Morpheus (SFT) | Self-learning mechanism |
| AI Safety v9 | Canary Head | Quality labeling via entropy |
| SNN-LM v4 | BitNet b1.58 | Energy-efficient inference |

## 📁 Repository Structure

```
snn-genesis/
├── experiments/
│   ├── phase1_snn_noise.py          # SNN random noise generation
│   ├── phase2_noise_injection.py    # LLM hidden state perturbation
│   ├── phase3_data_generation.py    # Dream Catcher v2 pipeline
│   ├── phase4_self_training.py      # QLoRA SFT self-improvement
│   └── phase5_evolution_loop.py     # Full pipeline integration
├── core/
│   ├── snn_reservoir.py             # Chaotic SNN reservoir (from Comprypto)
│   ├── canary_monitor.py            # Canary head entropy monitoring
│   └── quality_scorer.py            # Data quality evaluation
├── papers/
│   └── paper_genesis_v1.tex         # Paper draft
├── figures/
├── .gitignore
└── README.md
```

## 🚀 Quick Start

```bash
pip install torch transformers bitsandbytes peft snntorch
```

## 📬 Related Work

- [ANN-to-SNN Converter + AI Immune System (v11)](https://github.com/hafufu-stack/temporal-coding-simulation)
- [SNN-Comprypto](https://github.com/hafufu-stack/temporal-coding-simulation/tree/main/snn-comprypto)
- [SNN Language Model](https://github.com/hafufu-stack/snn-language-model)

## 📝 Citation

```bibtex
@misc{funasaki2026genesis,
  title={Project Genesis: Self-Evolving AI via SNN Chaotic Randomness},
  author={Funasaki, Hiroto},
  year={2026}
}
```

## 📜 License

MIT License
