# Autonomous Research Agent

A production-grade Python Autonomous Research Agent featuring a 4-node pipeline with an adversarial Critic and Qdrant evidence store.

## Architecture

See [docs/architecture.md](docs/architecture.md) for full architectural specifications.

```
Planner -> Researcher -> [Evidence Store: Qdrant] -> Analyst <--(Adversarial Red-Team)--> Critic
```

- **Planner** (`src/agents/planner.py`): Research goal decomposition & sub-task DAG creation.
- **Researcher** (`src/agents/researcher.py`): Document gathering & Qdrant evidence ingestion.
- **Analyst** (`src/agents/analyst.py`): Multi-source synthesis & report drafting with strict point-in-time evidence IDs.
- **Critic** (`src/agents/critic.py`): Adversarial evaluation engine auditing citations against Qdrant, challenging assumptions, and running revision loops.

## Tools & Evaluation
- **Financial Market API** (`src/tools/financial_api.py`): Quantitative ratios, historical prices, and news via `yfinance`.
- **Docling & Qdrant Store** (`src/tools/docling_qdrant.py`): Document parsing and vector retrieval.
- **Evaluation Harness** (`src/eval/llmops_harness.py`): Grounding precision, hallucination benchmarks, and adversarial red-team catch rates.

## Installation & Testing

```bash
# Install dependencies in editable mode
pip install -e .

# Run test suite
pytest tests/

# Launch Streamlit Research Dashboard
streamlit run src/ui/app.py
```
