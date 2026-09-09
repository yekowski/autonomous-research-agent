# Autonomous Research Agent

An experimental, multi-agent AI pipeline designed to automate financial research and value investment analysis. The system is built around the Mohnish Pabrai / Warren Buffett Dhandho framework and focuses on testing **closed-world knowledge enforcement** and **adversarial red-teaming**.

Instead of relying on an LLM's internal weights to generate analysis, this pipeline forces models to operate strictly on scraped, chunked, and vector-indexed regulatory filings and financial data. An adversarial Critic node audits the generated reports to ensure every claim is mathematically and semantically grounded in the retrieved evidence.

## 1. Problem Statement

Financial research requires high fidelity and zero hallucination. While LLMs are capable of synthesizing qualitative data, they frequently hallucinate numbers, misattribute metrics, or make ungrounded bullish claims. This project explores whether a multi-agent architecture with strict adversarial gating can force an LLM to produce a rigorous, strictly cited investment thesis without introducing external assumptions.

## 2. Architecture and Research Workflow

The system implements a 4-node Directed Acyclic Graph (DAG) workflow:

1. **Planner (`src/agents/planner.py`)**: Decomposes a target company query into sub-tasks (e.g., fetch 10-K, scrape historical prices, query fundamental tables).
2. **Researcher (`src/agents/researcher.py`)**: Executes tasks using `yfinance` and `edgartools`, processes raw PDFs using Docling, and indexes the resulting chunks into a local Qdrant vector database.
3. **Analyst (`src/agents/analyst.py`)**: Drafts a 4-part Dhandho investment report (Moat, Owner Earnings, Capital Allocation, Margin of Safety) using a split top-K retrieval strategy (pulling both semantic text and structured tables). It is heavily prompted to cite every claim using specific `[evidence_id]` tags.
4. **Critic (`src/agents/critic.py`)**: An adversarial red-team auditor. It uses regex and payload verification against the Qdrant store to ensure:
   - Every citation resolves to a valid chunk.
   - Assertions of a "Moat" are backed by ROIC/ROCE data.
   - Assertions of "Downside Protection" are backed by debt structure data.
   - The draft contains no empty/error payloads.
   If the Critic finds discrepancies, it rejects the draft and forces the Analyst into a revision loop (max 3 loops).

## 3. Key Engineering Decisions

* **Docling for PDF Parsing**: Standard text extraction destroys financial tables. Docling was chosen to preserve markdown table structures, which are tagged with `chunk_type="table"` and prioritized during Qdrant retrieval.
* **Closed-World Prompting**: The Analyst is explicitly instructed to output "Data Unavailable" rather than hallucinate missing metrics. 
* **Strict Citation Resolution**: Citations must map to exact 8-36 character UUIDs. The Critic strictly enforces `startswith` matching rather than `contains` to prevent ambiguous ID resolution.
* **Adversarial Loop Cap**: The Critic-Analyst loop is capped at 3 revisions to prevent infinite context expansion and timeout exhaustion, returning the highest-scoring draft if the cap is hit.

## 4. Implemented Capabilities

* **Multi-LLM Support**: Configured to run with Gemini API or locally via Ollama (e.g., `llama3.1:8b`, `qwen3:14b`).
* **Automated Data Pipeline**: Integrates SEC EDGAR fetching and fallback heuristic web hunting for international annual reports.
* **Evaluation Harness**: A built-in LLMOps harness (`src/eval/llmops_harness.py`) that tracks end-to-end latency, per-node latency, circuit breaking, and grounding rate against predefined SLAs.
* **Streamlit UI**: A visual dashboard (`src/ui/app.py`) for tracking node execution, monitoring SLA timeouts, and viewing Critic audit logs.

## 5. Testing and Evaluation Methodology

The system is tested using a suite of 34 pytest scaffolding scenarios (`tests/test_scaffold.py`).
* **Adversarial Citation Tests**: Evaluates whether the Critic successfully catches fabricated citations, hallucinatory claims, and unbacked moat assertions.
* **Valuation Audit Tests**: Ensures the Critic rejects reports that claim downside protection without citing debt data.
* **Pipeline Integration**: The UI and Harness allow for manual end-to-end runs on targets like "Transcorp Group" or "Beta Glass Plc", measuring total latency against a configurable SLA (e.g., 600s for local models).

## 6. Actual Results and Observations

* **Formatting Limitations of Smaller Models**: 3B parameter models (e.g., `llama3.2:3b`) consistently failed to adhere to the complex section-header and citation syntax requirements, leading to regex parse failures.
* **Context Window Bottlenecks**: When processing dense Docling-extracted financial tables, local 14B models (`qwen3:14b` on an 18GB M3 Pro) exhibit slow Time-To-First-Token (TTFT) and generation speeds, often breaching the 600-second SLA if forced into multiple adversarial revision loops.
* **Grounding Success**: When paired with a capable model (8B+ or Gemini API) and properly filtered table chunks, the Critic successfully forces the Analyst to cite explicit Qdrant UUIDs, significantly reducing hallucinated financial metrics.

## 7. Limitations and Known Failure Modes

* **Compute Intensive**: Running the full 4-node pipeline with 3 revision loops locally requires significant VRAM and time. The system will frequently SLA-timeout on consumer hardware if the LLM struggles to resolve Critic feedback quickly.
* **Extraction Fragility**: If `edgartools` or the PDF hunter fails to find the correct document, the Researcher returns empty payloads. The Analyst correctly outputs "Data Unavailable", but the pipeline cannot currently autonomously correct search strategy failures mid-run.
* **Table Context Bloat**: Feeding raw markdown tables into the LLM context window rapidly consumes tokens, leaving less room for analytical reasoning and increasing latency.
* **Semantic Verification Gap**: While the Critic verifies numeric extraction and heuristic contradictions, it relies on simple regex checks and keyword matching rather than a secondary LLM for deeper semantic verification of the citations.

## 8. Setup and Usage

**Prerequisites:** Python 3.13+, Qdrant (local or Docker), and Ollama (for local inference).

```bash
# Install dependencies
pip install -r requirements.txt

# Start Streamlit UI
streamlit run src/ui/app.py
```

**Environment Variables:**
* `LLM_PROVIDER`: Set to `ollama` or `gemini`.
* `GEMINI_API_KEY`: Required if using Gemini.
* `OLLAMA_MODEL`: Specify the local model (e.g., `llama3.1`).

## 9. Example Research Output/Workflow

1. **User Input**: `MTN Nigeria`
2. **Planner**: Creates DAG to fetch SEC filings and Yahoo Finance historical data.
3. **Researcher**: Downloads Annual Report, chunks via Docling, indexes to Qdrant.
4. **Analyst (Draft 1)**: Generates Dhandho thesis but forgets to cite Debt/Equity ratio.
5. **Critic (Loop 1)**: Rejects draft (`REVISE`). Flags Downside Protection section for missing balance sheet evidence.
6. **Analyst (Draft 2)**: Re-queries Qdrant for "Total Liabilities", updates report with citation `[b2a7...f1e9]`.
7. **Critic (Loop 2)**: Verifies citation exists and matches text. Returns `PASS`.
8. **Output**: Final markdown report presented in UI.

## 10. Future Work

* Implement a more robust secondary LLM inside the Critic for deeper semantic contradiction checks (moving beyond regex).
* Add support for more granular financial data providers (e.g., FMP, AlphaVantage) to reduce reliance on PDF table parsing.
* Implement a streaming UI to surface Analyst generation tokens in real-time, reducing perceived latency during long adversarial loops.
* Optimize Qdrant retrieval to summarize dense tables prior to injecting them into the Analyst's context window.
