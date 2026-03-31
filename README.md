# PSIP — Pipeline Security & Integrity Platform

[![CI](https://github.com/bp198/strategos-pipeline-defense/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/strategos-pipeline-defense/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Physics-Informed Game-Theoretic Defense of Pipeline Infrastructure**

MSc Thesis — Babak Pirzadi, STRATEGOS (2025)

---

## What this is

PSIP is a decision-support platform for pipeline security planners and integrity engineers. It combines four computation engines into a single REST API:

| Engine | Standard | What it computes |
|---|---|---|
| **FAD** | BS 7910:2019 Level 2 | Weld flaw acceptability (Kr, Lr, reserve factor) |
| **Monte Carlo** | PHMSA-calibrated | Failure probability P_f per segment (10,000 trials) |
| **Stackelberg Game** | Bayesian SSE / DOBSS | Optimal defender coverage allocation |
| **Adversarial** | FGSM / BIM / PGD | Attack success rate on WeldDefectMLP classifier |

Key results from the thesis: P_f range [0.29, 0.93], **17.0% risk reduction** vs. uniform allocation at B=0.40, PGD attack success rate 9.4%.

---

## Quick start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/strategos-pipeline-defense.git
cd strategos-pipeline-defense
pip install -e ".[dev]"

# Run the API locally
python main.py
# → open http://localhost:8000/docs

# Or with Docker (one command)
docker-compose up --build
# → open http://localhost:8000/docs
```

---

## Run the tests

```bash
pytest tests/
# 310 passed in ~8s
```

---

## Project structure

```
psip/           ← installable Python package (public API)
  fad/          ← BS 7910 FAD engine
  mc/           ← Monte Carlo P_f engine
  game/         ← Bayesian Stackelberg game engine
  nde/          ← WeldDefectMLP classifier
  adversarial/  ← FGSM / BIM / PGD attacks
  network/      ← Pipeline graph model
  fatigue/      ← IIW S-N fatigue engine
  api/          ← FastAPI REST API (all engines over HTTP)
src/            ← Original thesis engine implementations
tests/          ← 310 unit tests (pytest)
manuscript/     ← LaTeX thesis (66 pages, compiled PDF)
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Service health check |
| POST | `/api/fad/assess` | BS 7910 FAD assessment |
| POST | `/api/mc/simulate` | Monte Carlo P_f simulation |
| POST | `/api/game/solve` | Stackelberg Security Equilibrium |
| POST | `/api/adversarial/attack` | Adversarial attack on NDE classifier |
| GET | `/api/network/summary` | Pipeline network topology + P_f |

Full interactive documentation: `http://localhost:8000/docs`

---

## Author

**Babak Pirzadi** — babak.pirzadi@gmail.com
