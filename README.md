# AgenticPdmDataCleaner

_Agentic, LLM‑powered cleaning of automotive maintenance logs — with synthetic data, controlled noise injection, and repeatable benchmarks._

This repository provides a sandbox environment for AI agents focused on predictive maintenance. It supports generating synthetic automotive maintenance logs, injecting configurable noise, and benchmarking LLM agents that clean, repair, or reject log records using structured actions. The log schema is intentionally simplified for experimentation and reproducibility.

> TL;DR: Generate a synthetic fleet, corrupt the maintenance log with configurable noise (M1–M6), and evaluate multiple LLMs (OpenRouter / OpenAI) in a **stream-style, one‑record‑at‑a‑time** pipeline.

---

## Highlights

- **Synthetic Fleet Generator** → Fleet registry, odometer signals, service catalog, and maintenance logs.
- **Noise Taxonomy (M0–M6)** → Typos, missing values, out‑of‑fleet vehicles, wrong end dates, and more.
- **Agentic Benchmark** → LLMs choose one of: `accept(wo_num)`, `reject(wo_num)`, or `update(wo_num, field, value)`.
- **Config‑driven** → Swap models, fleets, and experiment knobs via YAML.
- **Reproducible** → Deterministic seeds & structured outputs ready for analysis.

Key folders:
- `agentic_pdm_data_cleaning/` – source code (agents, generators, evaluation, utilities).
- `bin/` – entry scripts (e.g., `run_experiment.py`).
- `config/` – app, model, fleet, benchmark, and hyper‑parameters YAMLs.
- `resources/` – static assets (e.g., service catalog).

---

## Quickstart (5 minutes)

### 1) Create a Python 3.12 virtual environment (recommended with `uv`)

```bash
uv venv --python 3.12
source .venv/bin/activate  
uv pip install --upgrade pip
```

> If you don't have `uv` installed, use `pip install uv` or fall back to the standard Python venv commands.

### 2) Ensure the Python path is set correctly

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### 3) Install dependencies

```bash
uv pip install -r requirements.txt
```

> If you hit a missing import later, just `pip install <that-package>` — the codebase is modular and you can add dependencies incrementally.

### 4) Set API keys (for model providers)
The benchmark configs use OpenRouter/OpenAI identifiers. Set one of:
```bash
export OPENROUTER_API_KEY="sk-..."
# or
export OPENAI_API_KEY="sk-..."
```
> Models are configured in `config/models/*.yaml`. Edit `full_model_identifier` or provider if you’re using local models or different endpoints.

### 5) Run the benchmark
```bash
python bin/run_experiment.py   --benchmark_config_path config/benchmarks/benchmark_ubix.yaml
```

Flags (all optional, defaults shown above):
- `--benchmark_config_path`: which suite to run (see `config/benchmarks/`).
- `--skip_data_generation`: reuse previously generated synthetic data.
- `--skip_benchmark_execution`: only generate data, skip LLM runs.

Outputs are written under `data/` and `logs/` (see `config/app_config.yaml`).

---

## Configuration Guide

- **App settings**: `config/app_config.yaml` (paths, logging).
- **Models**: `config/models/*.yaml` — provider (`open_router`), price metadata, and IDs like `openai/gpt-oss-20b`, `openai/gpt-5`.
- **Fleet**: `config/fleets/fleet_ubix.yaml` — country, monitoring window, fleet size.
- **Hyper‑parameters**: `config/hyperparameters_space/space_1.yaml` — decoding knobs such as `temperature`, `top_p`.
- **Benchmarks**: `config/benchmarks/*.yaml` — which experiments to run, how many records, and which models.

### Noise Categories (M0–M6)
- **M0**: clean
- **M1**: vehicle identifier misalignment
- **M2**: out‑of‑fleet vehicles
- **M3**: invalid categorical values (typos, swaps)
- **M4**: missing values
- **M5**: digital system test entries
- **M6**: wrong end dates (temporal inconsistency)

---

## How It Works (One‑Record Stream Processing)

For each maintenance record:
1. The agent optionally queries read‑only tools over **Fleet Registry**, **Odometer Signals**, and **Service Catalog**.
2. It must choose **exactly one** action:
   - `accept(wo_num)` — record is valid;
   - `reject(wo_num)` — irreparable or out‑of‑scope;
   - `update(wo_num, field, value)` — single‑field repair.
3. Results are saved in structured form for evaluation.

This mirrors how an online PdM pipeline would curate logs as they arrive.

---

## Reproducing the Paper’s Experiments

- Use the provided benchmark files in `config/benchmarks/`.
- Models are declared in each benchmark YAML under `models: [...]`.
- The code logs token usage, latency, and cost estimates (from model config files).
- After runs complete, evaluation scripts parse outputs and compute **EDR/ECR** per noise type.

> Tip: Start with a single small model (e.g., `nemotron-nano-9b-v2`) and `num_records: 10` to smoke‑test the pipeline, then scale up.

---

## FAQ

**Q: Do I need real data?**  
No. The framework generates **synthetic, privacy‑safe** data across four tables and then injects configurable noise into maintenance logs.

**Q: Can I plug in my own LLM?**  
Yes — add a YAML under `config/models/` with your provider and identifier, then include it in a benchmark file.

**Example:**
```yaml
model_name: "llama3.1:latest"
organization: "meta"
full_model_identifier: "meta/llama3.1:latest"
llm_config:
  provider: "open_ai"
  base_url: "http://private-server:11434/v1"
  cost_in: 0.01
  cost_out: 0.01
  cost_unit: 1_000_000
```


**Q: Where are results stored?**  
See `config/app_config.yaml`: artifacts go under `data/` and logs under `logs/` by default.

---

## License

MIT License

Copyright (c) 2024 vdimidov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
