# Skills Configuration

This document defines the agent skill interfaces, tool bindings, and operational contracts available across the Autonomous Research Agent pipeline.

---

## 1. financial_data_scraper

- **Identifier**: `financial_data_scraper`
- **Module Binding**: [`src/tools/financial_api.py`](file:///Users/yekowski/agy2-projects/autonomous_research_agent/src/tools/financial_api.py)
- **Target Class**: `FinancialDataTool`
- **Description**: Scrapes quantitative metrics and historical pricing via yfinance.

### Capabilities & Exposed Methods
1. `get_ticker_summary(ticker: str) -> Dict[str, Any]`
   - Extracts company profile, sector, industry classification, market capitalization, trailing P/E, forward P/E, and 52-week trading bands.
2. `get_financial_statements(ticker: str, statement_type: str = "income", quarterly: bool = False) -> Dict[str, Any]`
   - Retrieves GAAP/IFRS financial reports (`income`, `balance_sheet`, or `cashflow`) on an annual or quarterly basis.
3. `get_historical_prices(ticker: str, period: str = "1y", interval: str = "1d") -> List[Dict[str, Any]]`
   - Fetches historical Open, High, Low, Close, Volume (OHLCV) time series data for trend, moving average, and volatility calculations.
4. `get_news(ticker: str) -> List[Dict[str, Any]]`
   - Gathers latest corporate press releases, news headlines, publisher metadata, and article links.

### Usage Constraints
- All ticker symbols must be uppercase string identifiers (e.g., `AAPL`, `MSFT`, `NVDA`).
- Quarterly queries should be utilized when evaluating trailing sequential trends or recent earnings surprises.
- Numerical outputs must preserve native precision (do not round prematurely before downstream analysis).

---

## 2. vector_evidence_retriever

- **Identifier**: `vector_evidence_retriever`
- **Module Binding**: [`src/tools/docling_qdrant.py`](file:///Users/yekowski/agy2-projects/autonomous_research_agent/src/tools/docling_qdrant.py)
- **Target Class**: `DoclingQdrantStore`
- **Description**: Parses SEC filings/PDFs and queries the Qdrant evidence store.

### Capabilities & Exposed Methods
1. `parse_document(document_path_or_url: str) -> List[Dict[str, Any]]`
   - Utilizes Docling to ingest PDFs, 10-K/10-Q SEC disclosures, earnings call transcripts, and investor presentations, preserving tabular geometry and hierarchical sections.
2. `ingest_evidence(chunks: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None) -> List[str]`
   - Chunks and stores parsed text elements with rich metadata into the Qdrant vector collection. Returns immutable `evidence_id` UUIDs.
3. `search_evidence(query_vector: List[float], top_k: int = 5, source_filter: Optional[str] = None) -> List[Dict[str, Any]]`
   - Performs cosine vector similarity search against the indexed evidence store to retrieve substantiated context chunks.
4. `retrieve_by_id(evidence_id: str) -> Optional[Dict[str, Any]]`
   - Fetches a specific evidence point payload to enable deterministic citation verification by the Critic node.

### Evidence Payload Contract
Each point stored in Qdrant strictly complies with the following payload structure:
```json
{
  "evidence_id": "urn:uuid:<uuid_v4>",
  "content": "<parsed text content>",
  "source": "<filename, URL, or filing identifier>",
  "page": 1,
  "metadata": {
    "section_title": "<Item 7. MD&A, etc.>",
    "filing_date": "<YYYY-MM-DD>"
  }
}
```

---

## 3. fetch_company_filings

- **Identifier**: `fetch_company_filings`
- **Module Binding**: [`src/tools/docling_qdrant.py`](file:///Users/yekowski/agy2-projects/autonomous_research_agent/src/tools/docling_qdrant.py)
- **Target Class**: `DoclingQdrantStore`
- **Description**: A market-aware jurisdiction router. Automatically fetches global regulatory filings (like SEC 10-Ks for US equities) based on the ticker suffix, and gracefully degrades for international markets.

### Capabilities & Exposed Methods
1. `fetch_company_filings(ticker: str, form_type: str = "10-K") -> List[str]`
   - Configures SEC User-Agent compliance identity via `set_identity("researcher@autonomous.agent")`.
   - Evaluates ticker jurisdiction:
     - For US equities (no suffix, e.g. `AAPL`, `MSFT`), fetches latest filing via `Company(ticker).get_filings(form=form_type).latest()`, chunks markdown/tables, and indexes into Qdrant.
     - For non-US equities ending with `.LG` (NGX), `.L` (LSE), or `.TO` (TSX), gracefully bypasses SEC EDGAR, returns `[]`, and logs a jurisdiction warning so downstream nodes rely on fundamental API data.
   - Returns the list of indexed `evidence_id` UUIDs.

### Usage Constraints
- Universal filing acquisition tool for all tickers across domestic and international markets.
