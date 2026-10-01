# Otún-LM

A language model focused on Colombian Spanish, built from scratch in phases.
Named after the Otún river in Pereira, Colombia.

Models follow the naming scheme `Otún-124M`, `Otún-7B`, etc.

## Phase 1 — Foundations (8 weeks)

Based on Andrej Karpathy's *Neural Networks: Zero to Hero*, adapted to Spanish.

| Week | Module | Description |
|------|--------|-------------|
| 1 | `01-micrograd` | Scalar autograd engine and backpropagation from scratch |
| 2 | `02-makemore-bigram-mlp` | Character-level models on Colombian municipality names and surnames |
| 3 | `03-makemore-activations-backprop` | Activations, BatchNorm, manual backprop |
| 4 | `04-gpt-from-scratch` | Transformer trained on *María* (Jorge Isaacs) |
| 5 | `05-tokenizer` | Custom BPE tokenizer for Spanish; efficiency vs. GPT-2 tokenizer |
| 6 | `06-data-pipeline` | Spanish Wikipedia + FineWeb-2 subset, remote GPU, W&B |
| 7 | `07-gpt2-124m-spanish` | Otún-124M: GPT-2-sized model trained on Spanish |
| 8 | `08-eval-writeup` | Checkpoint evaluation and public write-up |

## Repository layout

```
lab-notebook/   Experiment log (Spanish), one entry per experiment
phase1/         Code for each Phase 1 module
data/           Downloaded datasets (gitignored; only download scripts are tracked)
writeups/       Public technical write-ups (English)
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## License

TBD.
