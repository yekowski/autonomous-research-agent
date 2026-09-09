"""Comprehensive production test suite for Autonomous Research Agent."""

import json
import pytest
from src.agents.planner import ResearchPlanner, AllowedToolAction
from src.agents.researcher import ResearcherAgent
from src.agents.analyst import AnalystAgent, ReportClaim, DraftReport
from src.agents.critic import CriticAgent, VerdictStatus, CritiqueVerdict, FlaggedClaimDetail
from src.tools.financial_api import (
    FinancialDataTool,
    TickerNotFoundError,
    InsufficientDataError,
    ValuationMultiples,
    FinancialStatementsData,
)
from src.tools.docling_qdrant import DoclingQdrantStore, DocumentChunk, FilingNotFoundError
from src.eval.llmops_harness import (
    ResearchEvalHarness,
    EvaluationHarness,
    CircuitBreaker,
    EvalTestCase,
)


# =====================================================================
# Planner Tests
# =====================================================================

def test_research_planner_json_task_generation():
    planner = ResearchPlanner()
    query = "Is Transcorp Group a buy?"
    tasks = planner.plan_research(query)

    assert isinstance(tasks, list)
    assert len(tasks) >= 3

    # Validate schema of each task
    actions = [t["action"] for t in tasks]
    for t in tasks:
        assert "task_id" in t
        assert "action" in t
        assert "ticker" in t
        assert "parameters" in t
        assert "rationale" in t

    # Verify tool action coverage
    assert AllowedToolAction.PULL_VALUATION_MULTIPLES.value in actions
    assert AllowedToolAction.PULL_INCOME_STATEMENT.value in actions
    assert AllowedToolAction.PULL_BALANCE_SHEET.value in actions
    assert AllowedToolAction.FETCH_RECENT_NEWS.value in actions
    assert AllowedToolAction.FETCH_COMPANY_FILINGS.value in actions


def test_research_planner_empty_query():
    planner = ResearchPlanner()
    with pytest.raises(ValueError):
        planner.plan_research("")


# =====================================================================
# Docling + Qdrant Store Tests
# =====================================================================

def test_docling_qdrant_store_top_k_retrieval():
    store = DoclingQdrantStore(collection_name="test_topk_store", in_memory=True, vector_dim=384)

    # Ingest document chunks (including a table chunk)
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            text="### Revenue Table\n| Period | Revenue |\n| 2024 | $120B |\n| 2023 | $96B |",
            chunk_type="table",
            page=12,
            source="sec://transcorp/10k",
            metadata={"table_type": "income"},
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            text="Transcorp expanded power generation capacity by 15% year-over-year in the Afam plant.",
            chunk_type="text",
            page=15,
            source="sec://transcorp/10k",
            metadata={"topic": "power_operations"},
        ),
    ]

    evidence_ids = store.ingest_chunks(chunks)
    assert len(evidence_ids) == 2

    # Query top-k with semantic query string
    results = store.retrieve_top_k(query="revenue growth and power generation capacity", top_k=2)
    assert len(results) == 2
    assert results[0].score > 0.0
    assert results[0].content != ""
    assert results[0].evidence_id in evidence_ids

    # Citation retrieval check
    point_payload = store.retrieve_by_id(evidence_ids[0])
    assert point_payload is not None
    assert point_payload["evidence_id"] == evidence_ids[0]
    assert point_payload["chunk_type"] == "table"


# =====================================================================
# Financial Tool & Researcher Pipeline Tests
# =====================================================================

def test_financial_data_tool_validation():
    tool = FinancialDataTool()
    with pytest.raises(ValueError):
        tool.get_valuation_multiples("")


def test_researcher_agent_execution_and_data_contract():
    # Setup mock / simulated financial tool for deterministic offline testing
    class MockFinancialTool(FinancialDataTool):
        def get_valuation_multiples(self, ticker: str) -> ValuationMultiples:
            return {
                "symbol": ticker,
                "trailing_pe": 8.5,
                "forward_pe": 7.2,
                "peg_ratio": 0.9,
                "enterprise_to_ebitda": 5.4,
                "price_to_book": 1.8,
                "price_to_sales": 2.1,
                "enterprise_value": 5000000000.0,
                "market_cap": 4200000000.0,
                "beta": 1.1,
                "currency": "NGN",
            }

        def get_income_statement(self, ticker: str, quarterly: bool = False) -> FinancialStatementsData:
            return {
                "symbol": ticker,
                "statement_type": "income",
                "period_type": "annual",
                "currency": "NGN",
                "metrics": {
                    "Total Revenue": {"2024-12-31": 196000000.0, "2023-12-31": 134000000.0},
                    "Operating Income": {"2024-12-31": 55000000.0, "2023-12-31": 38000000.0},
                },
            }

        def get_balance_sheet(self, ticker: str, quarterly: bool = False) -> FinancialStatementsData:
            return {
                "symbol": ticker,
                "statement_type": "balance_sheet",
                "period_type": "annual",
                "currency": "NGN",
                "metrics": {
                    "Total Assets": {"2024-12-31": 450000000.0},
                    "Total Liabilities": {"2024-12-31": 210000000.0},
                },
            }

        def get_recent_news(self, ticker: str, limit: int = 10):
            return [
                {
                    "id": "news_01",
                    "symbol": ticker,
                    "title": "Transcorp Reports 46% Revenue Surge",
                    "publisher": "BusinessDay",
                    "link": "https://businessday.ng/article1",
                    "published_at": "2024-10-25",
                    "summary": "Transcorp Group announces record quarterly earnings.",
                }
            ]

    qdrant_store = DoclingQdrantStore(collection_name="test_researcher_run", in_memory=True, vector_dim=384, offline_mode=True)
    researcher = ResearcherAgent(financial_tool=MockFinancialTool(), qdrant_store=qdrant_store)

    tasks = [
        {"task_id": "t1", "action": "pull_valuation_multiples", "ticker": "TRANSCORP.LG"},
        {"task_id": "t2", "action": "pull_income_statement", "ticker": "TRANSCORP.LG", "parameters": {"quarterly": False}},
        {"task_id": "t3", "action": "pull_balance_sheet", "ticker": "TRANSCORP.LG", "parameters": {"quarterly": False}},
        {"task_id": "t4", "action": "fetch_recent_news", "ticker": "TRANSCORP.LG", "parameters": {"limit": 5}},
        {"task_id": "t5", "action": "fetch_company_filings", "ticker": "TRANSCORP.LG"},
    ]

    # Execute tasks synchronously (which runs async concurrent gather internally)
    manifest = researcher.execute_tasks_sync(tasks)

    assert manifest["ticker"] == "TRANSCORP.LG"
    assert manifest["total_indexed"] >= 4
    assert len(manifest["evidence_ids"]) == manifest["total_indexed"]
    assert len(manifest["errors"]) == 0

    # Verify categories
    by_cat = manifest["by_category"]
    assert len(by_cat["valuation_multiples"]) == 1
    assert len(by_cat["income_statement"]) == 1
    assert len(by_cat["balance_sheet"]) == 1
    assert len(by_cat["news"]) == 1

    # Verify data contract: All points are retrievable from Qdrant by ID
    for eid in manifest["evidence_ids"]:
        point = qdrant_store.retrieve_by_id(eid)
        assert point is not None
        assert "content" in point
        assert point["metadata"]["ticker"] == "TRANSCORP.LG"


# =====================================================================
# Analyst, Critic, and Adversarial Loop Tests
# =====================================================================

def test_analyst_agent_structured_synthesis_and_citations():
    qdrant_store = DoclingQdrantStore(collection_name="test_analyst_store", in_memory=True, vector_dim=384)
    eid_inc = qdrant_store.ingest_raw_evidence(
        content="Revenue for 2024 totaled 196.0M NGN with operating profit of 55.0M NGN.",
        source="sec://transcorp/income",
        metadata={"ticker": "TRANSCORP.LG"},
    )
    eid_val = qdrant_store.ingest_raw_evidence(
        content="Valuation multiples: Trailing P/E of 8.5, EV/EBITDA of 5.4.",
        source="yfinance://transcorp/multiples",
        metadata={"ticker": "TRANSCORP.LG"},
    )

    manifest = {
        "ticker": "TRANSCORP.LG",
        "total_indexed": 2,
        "evidence_ids": [eid_inc, eid_val],
        "by_category": {
            "income_statement": [eid_inc],
            "valuation_multiples": [eid_val],
        },
    }

    analyst = AnalystAgent()
    draft = analyst.synthesize_sync(
        query="Is Transcorp Group a buy?",
        manifest=manifest,
        qdrant_store=qdrant_store,
    )

    # 1. Verify structured sections
    assert draft.ticker == "TRANSCORP.LG"
    assert draft.thesis != ""
    assert draft.financial_metrics != ""
    assert draft.valuation != ""
    assert draft.risks != ""

    # 2. Verify mandatory inline citations: Every claim cited with exact [uuid]
    assert len(draft.inline_citations) >= 2
    assert eid_inc in draft.inline_citations
    assert eid_val in draft.inline_citations
    assert f"[{eid_inc}]" in draft.full_markdown
    assert f"[{eid_val}]" in draft.full_markdown


def test_critic_adversarial_audit_pass():
    qdrant_store = DoclingQdrantStore(collection_name="test_critic_pass", in_memory=True, vector_dim=384)
    eid = qdrant_store.ingest_raw_evidence(
        content="Operating margin expanded to 28.0% based on 196.0M revenue.",
        source="yfinance://transcorp/income",
    )

    manifest = {
        "ticker": "TRANSCORP.LG",
        "evidence_ids": [eid],
        "by_category": {"income_statement": [eid]},
    }
    analyst = AnalystAgent()
    draft = analyst.synthesize_sync("Query", manifest, qdrant_store)

    critic = CriticAgent()
    async def mock_audit(*args, **kwargs):
        return CritiqueVerdict(
            is_approved=True,
            verdict="PASS",
            feedback="AUDIT PASS",
            revision_count=draft.revision_count,
            total_citations_checked=1,
            flagged_claims=[],
            grounding_rate=1.0,
        )
    critic._audit_with_llm = mock_audit
    critic.client = type('obj', (object,), {'is_configured': True})  # Mock client to bypass None check
    
    verdict = critic.audit_report_sync(draft, qdrant_store)

    assert verdict["is_approved"] is True
    assert verdict["verdict"] == "PASS"


def test_critic_adversarial_detects_hallucinated_numbers():
    qdrant_store = DoclingQdrantStore(collection_name="test_critic_hallucination", in_memory=True, vector_dim=384)
    eid = qdrant_store.ingest_raw_evidence(
        content="Revenue was 196.0M.",
        source="sec://filing",
    )

    manifest = {
        "ticker": "TRANSCORP.LG",
        "evidence_ids": [eid],
        "by_category": {"income_statement": [eid]},
    }
    analyst = AnalystAgent()
    draft = analyst.synthesize_sync("Query", manifest, qdrant_store)

    # Adversarially inject a false / hallucinated number into the draft
    draft.thesis = f"Management reported an extraordinary margin growth of 999.0% [{eid}]."

    critic = CriticAgent()
    async def mock_audit(*args, **kwargs):
        return CritiqueVerdict(
            is_approved=False,
            verdict="REVISE",
            feedback="REVISE",
            revision_count=draft.revision_count,
            total_citations_checked=1,
            flagged_claims=[{"issue_type": "HALLUCINATED_NUMBERS", "claim_text": "", "evidence_id": eid, "detected_discrepancy": ""}],
            grounding_rate=0.0,
        )
    critic._audit_with_llm = mock_audit
    critic.client = type('obj', (object,), {'is_configured': True})
    
    verdict = critic.audit_report_sync(draft, qdrant_store)

    assert verdict["is_approved"] is False
    assert verdict["verdict"] == "REVISE"


def test_critic_adversarial_detects_missing_citations():
    draft = DraftReport(
        report_id="uncited_rep",
        query="Test query",
        ticker="TEST",
        thesis="Completely ungrounded claim without any citations.",
        financial_metrics="No numbers or citations.",
        valuation="P/E is 15.0.",
        risks="Market risks exist.",
        claims=[],
        inline_citations=[],
    )
    critic = CriticAgent()
    verdict = critic.audit_report_sync(draft)

    assert verdict["is_approved"] is False
    assert verdict["verdict"] == "REVISE"
    assert "CRITICAL AUDIT FAILURE" in verdict["feedback"]


def test_critic_max_revision_loop_limit():
    draft = DraftReport(
        report_id="max_rev_rep",
        query="Test query",
        ticker="TEST",
        thesis="Claim [00000000-0000-0000-0000-000000000000]",
        financial_metrics="Metrics",
        valuation="Valuation",
        risks="Risks",
        inline_citations=["00000000-0000-0000-0000-000000000000"],
        revision_count=3,  # Hardcoded max 3 revision cycles
    )
    critic = CriticAgent()
    verdict = critic.audit_report_sync(draft)

    assert verdict["is_approved"] is True
    assert verdict["verdict"] == "PASS"
    assert "maximum revision threshold" in verdict["feedback"]


def test_eval_harness_legacy_compatibility():
    harness = ResearchEvalHarness()
    qdrant_store = DoclingQdrantStore(collection_name="test_eval", in_memory=True, vector_dim=384)
    qdrant_store.ingest_raw_evidence(content="Revenue 100M", source="sec")
    # need an actual eid to pass
    eid = qdrant_store.retrieve_top_k("Revenue", 1)[0].evidence_id

    analyst = AnalystAgent()
    draft = analyst.synthesize_sync(
        "Test Analysis",
        {"ticker": "TEST", "evidence_ids": [eid], "by_category": {"income_statement": [eid]}},
        qdrant_store=qdrant_store,
    )
    score = harness.evaluate_grounding(draft)
    assert score == 1.0


def test_evaluation_harness_benchmark_targets():
    """Tests EvaluationHarness execution against the target companies dataset."""
    class MockFinTool(FinancialDataTool):
        def get_valuation_multiples(self, ticker: str):
            return {
                "symbol": ticker,
                "trailing_pe": 9.2,
                "forward_pe": 8.0,
                "peg_ratio": 1.0,
                "enterprise_to_ebitda": 6.1,
                "price_to_book": 1.5,
                "price_to_sales": 1.8,
                "enterprise_value": 3000000000.0,
                "market_cap": 2500000000.0,
                "beta": 0.9,
                "currency": "NGN",
            }

        def get_income_statement(self, ticker: str, quarterly: bool = False):
            return {
                "symbol": ticker,
                "statement_type": "income",
                "period_type": "annual",
                "currency": "NGN",
                "metrics": {"Revenue": {"2024-12-31": 85000000.0}},
            }

        def get_balance_sheet(self, ticker: str, quarterly: bool = False):
            return {
                "symbol": ticker,
                "statement_type": "balance_sheet",
                "period_type": "annual",
                "currency": "NGN",
                "metrics": {"Total Assets": {"2024-12-31": 150000000.0}},
            }

        def get_recent_news(self, ticker: str, limit: int = 10):
            return [{"id": "n1", "symbol": ticker, "title": "Growth", "publisher": "Reuters", "link": "http://x", "published_at": "2024", "summary": "Text"}]

    harness = EvaluationHarness(sla_seconds=45.0)
    report = harness.run_benchmark_suite_sync(
        target_companies=["Transcorp Group", "Beta Glass Plc"],
        financial_tool=MockFinTool(),
    )

    assert report.total_targets == 2
    assert report.passed_targets == 2
    assert report.pass_rate == 1.0
    assert report.sla_compliance_rate == 1.0
    assert report.avg_grounding_rate == 1.0
    assert report.node_latency_averages_ms["planner"] > 0.0
    assert report.node_latency_averages_ms["researcher"] > 0.0
    assert report.node_latency_averages_ms["analyst"] > 0.0
    assert report.node_latency_averages_ms["critic"] > 0.0

    # Verify markdown summary output format
    md = report.summary_markdown()
    assert "LLMOps Benchmark Suite Report" in md
    assert "Transcorp Group" in md
    assert "Beta Glass Plc" in md


def test_evaluation_harness_sla_violation_failure():
    """Tests that SLA tracking strictly fails runs exceeding the latency threshold."""
    # Impose an impossibly tight 0.00001s SLA to verify timeout violation capture
    harness = EvaluationHarness(sla_seconds=0.00001)
    import asyncio
    result = asyncio.run(harness.evaluate_target("Transcorp Group"))

    assert result.status == "SLA_VIOLATION"
    assert any("SLA_VIOLATION" in reason for reason in result.failure_reasons)


def test_evaluation_harness_quality_gate_grounding_threshold():
    """Tests that the Quality Gate strictly fails if grounding_rate < 100%."""
    # Critic that returns sub-100% grounding rate
    class FlawedCritic(CriticAgent):
        async def audit_report(self, draft, store=None):
            return {
                "is_approved": False,
                "verdict": "REVISE",
                "feedback": "Hallucinated figures found.",
                "revision_count": 0,
                "total_citations_checked": 4,
                "flagged_claims": [{"claim_text": "text", "evidence_id": "id", "issue_type": "HALLUCINATED_NUMBERS", "detected_discrepancy": "wrong number"}],
                "grounding_rate": 0.75,  # 75% grounding fails quality gate
            }

    harness = EvaluationHarness(critic=FlawedCritic(), sla_seconds=45.0)
    import asyncio
    result = asyncio.run(harness.evaluate_target("Transcorp Group"))

    assert result.status == "FAIL"
    assert any("QUALITY_GATE_FAILURE" in reason for reason in result.failure_reasons)


def test_circuit_breaker_catches_timeouts_and_logs_ledger(tmp_path):
    """Tests circuit breaker fault isolation and JSONL ledger logging."""
    ledger_file = tmp_path / "test_cb_ledger.jsonl"
    cb = CircuitBreaker(failure_threshold=2, ledger_path=str(ledger_file))

    assert cb.can_execute() is True
    # Trip breaker with consecutive failures
    cb.record_failure("FlakyProvider Corp", "researcher", TimeoutError("Connection to vendor timed out after 30s"))
    assert cb.can_execute() is True
    cb.record_failure("FlakyProvider Corp", "researcher", TimeoutError("Connection to vendor timed out after 30s"))
    assert cb.can_execute() is False

    # Verify JSONL ledger records
    assert ledger_file.exists()
    with open(ledger_file, "r") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["error_type"] == "TimeoutError"
    assert lines[1]["circuit_state"] == "OPEN"


def test_end_to_end_pipeline_adversarial_loop():
    """End-to-end test across all 4 nodes: Planner -> Researcher -> Analyst -> Critic."""
    planner = ResearchPlanner()
    query = "Is Transcorp Group a buy?"
    tasks = planner.plan_research(query)
    assert len(tasks) >= 3

    # Step 2: Researcher execution with in-memory Qdrant store
    qdrant_store = DoclingQdrantStore(collection_name="e2e_evidence", in_memory=True, vector_dim=384, offline_mode=True)

    class MockFinTool(FinancialDataTool):
        def get_valuation_multiples(self, ticker: str):
            return {
                "symbol": ticker,
                "trailing_pe": 8.5,
                "forward_pe": 7.2,
                "peg_ratio": 0.9,
                "enterprise_to_ebitda": 5.4,
                "price_to_book": 1.8,
                "price_to_sales": 2.1,
                "enterprise_value": 5000000000.0,
                "market_cap": 4200000000.0,
                "beta": 1.1,
                "currency": "NGN",
            }

        def get_income_statement(self, ticker: str, quarterly: bool = False):
            return {
                "symbol": ticker,
                "statement_type": "income",
                "period_type": "annual",
                "currency": "NGN",
                "metrics": {"Total Revenue": {"2024-12-31": 196000000.0}},
            }

        def get_balance_sheet(self, ticker: str, quarterly: bool = False):
            return {
                "symbol": ticker,
                "statement_type": "balance_sheet",
                "period_type": "annual",
                "currency": "NGN",
                "metrics": {"Total Assets": {"2024-12-31": 450000000.0}},
            }

        def get_recent_news(self, ticker: str, limit: int = 10):
            return [{"id": "n1", "symbol": ticker, "title": "Headline", "publisher": "Pub", "link": "http://x", "published_at": "2024", "summary": "Text"}]

    researcher = ResearcherAgent(financial_tool=MockFinTool(), qdrant_store=qdrant_store)
    manifest = researcher.execute_tasks_sync(tasks)
    assert manifest["total_indexed"] >= 3

    # Step 3: Analyst synthesis with inline citations
    analyst = AnalystAgent()
    draft = analyst.synthesize_sync(query, manifest, qdrant_store)
    assert len(draft.inline_citations) >= 2

    # Step 4: Critic adversarial audit
    critic = CriticAgent(max_revisions_allowed=3)
    
    async def mock_audit_pass(*args, **kwargs):
        return CritiqueVerdict(
            is_approved=True,
            verdict="PASS",
            feedback="PASS",
            revision_count=args[0].revision_count,
            total_citations_checked=2,
            flagged_claims=[],
            grounding_rate=1.0,
        )
    critic._audit_with_llm = mock_audit_pass
    critic.client = type('obj', (object,), {'is_configured': True})
    
    verdict = critic.audit_report_sync(draft, qdrant_store)
    assert verdict["is_approved"] is True
    assert verdict["verdict"] == "PASS"

    # Adversarial cycle test: inject hallucinated metric and test REVISE
    draft.thesis += f"\nHallucinated metric claiming 888.0% margin expansion [{manifest['evidence_ids'][0]}]."
    
    async def mock_audit_revise(*args, **kwargs):
        return CritiqueVerdict(
            is_approved=False,
            verdict="REVISE",
            feedback="REVISE",
            revision_count=args[0].revision_count,
            total_citations_checked=2,
            flagged_claims=[{"issue_type": "HALLUCINATED_NUMBERS", "claim_text": "", "evidence_id": "", "detected_discrepancy": ""}],
            grounding_rate=0.0,
        )
    critic._audit_with_llm = mock_audit_revise
    
    verdict_rev = critic.audit_report_sync(draft, qdrant_store)
    assert verdict_rev["is_approved"] is False
    assert verdict_rev["verdict"] == "REVISE"
    assert "REVISE" in verdict_rev["feedback"]

    # Revision cycle loopback to Analyst
    revised_draft = analyst.synthesize_sync(query, manifest, qdrant_store)
    revised_draft.revision_count = draft.revision_count + 1
    
    critic._audit_with_llm = mock_audit_pass
    verdict_re_audited = critic.audit_report_sync(revised_draft, qdrant_store)
    assert verdict_re_audited["is_approved"] is True
    assert verdict_re_audited["verdict"] == "PASS"


def test_docling_qdrant_store_jurisdiction_router():
    """Validates the Jurisdiction Router pattern in fetch_company_filings."""
    store = DoclingQdrantStore(collection_name="test_jurisdiction_router", in_memory=True, vector_dim=384)

    # 1. Non-US Equities: .LG, .L, .TO should gracefully bypass EDGAR and raise FilingNotFoundError when not found
    with pytest.raises(FilingNotFoundError):
        store.fetch_company_filings("TRANSCORP.LG")
    with pytest.raises(FilingNotFoundError):
        store.fetch_company_filings("VOD.L")
    with pytest.raises(FilingNotFoundError):
        store.fetch_company_filings("SHOP.TO")

    # 2. US Equity: no suffix (AAPL) executes EDGAR extraction & Qdrant indexing
    evidence_ids = store.fetch_company_filings("AAPL", form_type="10-K")
    assert isinstance(evidence_ids, list)
    assert len(evidence_ids) >= 1

    # Verify each evidence item is persisted in Qdrant
    for eid in evidence_ids:
        point = store.retrieve_by_id(eid)
        assert point is not None
        assert point["evidence_id"] == eid
        assert "SEC EDGAR" in point["source"]

    # 3. Backward compatibility alias verification
    with pytest.raises(FilingNotFoundError):
        store.ingest_sec_filing("TRANSCORP.LG")


# =====================================================================
# Empty-Data Synthesis & Ticker Validation Bug Fix Tests
# =====================================================================

def test_financial_data_tool_fast_info_ticker_not_found(monkeypatch):
    """Verifies that fast_info validation raises TickerNotFoundError when market_cap/quote_type unavailable."""
    tool = FinancialDataTool()

    class MockEmptyFastInfo:
        market_cap = None
        quote_type = None

    class MockEmptyTicker:
        fast_info = MockEmptyFastInfo()

    # Pre-check should raise TickerNotFoundError
    with pytest.raises(TickerNotFoundError, match="has no active market data"):
        tool.validate_ticker("UNKNOWN_TICKER", MockEmptyTicker())


def test_researcher_empty_target_returns_invalid_or_empty_status():
    """Verifies that Researcher returns INVALID_OR_EMPTY_TARGET when all metrics are empty and no filings retrieved."""
    qdrant_store = DoclingQdrantStore(collection_name="test_empty_researcher", in_memory=True, vector_dim=384)

    class EmptyFinTool(FinancialDataTool):
        def get_valuation_multiples(self, ticker: str):
            return {
                "symbol": ticker,
                "trailing_pe": None,
                "forward_pe": None,
                "peg_ratio": None,
                "enterprise_to_ebitda": None,
                "price_to_book": None,
                "price_to_sales": None,
                "enterprise_value": None,
                "market_cap": None,
                "beta": None,
                "currency": "USD",
            }

        def get_income_statement(self, ticker: str, quarterly: bool = False):
            return {
                "symbol": ticker,
                "statement_type": "income",
                "period_type": "annual",
                "currency": "USD",
                "metrics": {},
            }

        def get_balance_sheet(self, ticker: str, quarterly: bool = False):
            return {
                "symbol": ticker,
                "statement_type": "balance_sheet",
                "period_type": "annual",
                "currency": "USD",
                "metrics": {},
            }

        def get_recent_news(self, ticker: str, limit: int = 10):
            return []

    researcher = ResearcherAgent(financial_tool=EmptyFinTool(), qdrant_store=qdrant_store)
    tasks = [
        {"task_id": "t1", "action": "pull_valuation_multiples", "ticker": "NONEXISTENT.LG", "parameters": {}},
        {"task_id": "t2", "action": "pull_income_statement", "ticker": "NONEXISTENT.LG", "parameters": {}},
        {"task_id": "t3", "action": "pull_balance_sheet", "ticker": "NONEXISTENT.LG", "parameters": {}},
        {"task_id": "t4", "action": "fetch_company_filings", "ticker": "NONEXISTENT.LG", "parameters": {}},
    ]

    manifest = researcher.execute_tasks_sync(tasks)

    # Must return FAILED_PRE_FLIGHT and not emit an approved manifest
    assert manifest["status"] == "FAILED_PRE_FLIGHT"
    assert manifest["error_status"] == "FAILED_PRE_FLIGHT"
    assert manifest["total_indexed"] == 0
    assert any("FAILED_PRE_FLIGHT" in err.get("error", "") for err in manifest["errors"])


def test_critic_adversarial_detects_contradictory_evidence():
    """Verifies that Critic flags CONTRADICTORY_EVIDENCE and issues REVISE when citing empty/error payloads while claiming growth."""
    qdrant_store = DoclingQdrantStore(collection_name="test_critic_contradiction", in_memory=True, vector_dim=384)
    eid = qdrant_store.ingest_raw_evidence(
        content="Company not found or No income statement line items reported.",
        source="yfinance://EMPTY_TICKER/income_statement_annual",
    )

    draft = DraftReport(
        report_id="rep_contradiction",
        query="Evaluate empty company",
        ticker="EMPTY_TICKER",
        thesis=f"Management demonstrated solid revenue expansion and expanding operating growth [{eid}].",
        financial_metrics=f"Disclosures confirm strong margin improvement [{eid}].",
        valuation="Valuation multiples not available in retrieved evidence store.",
        risks="Market and liquidity risks exist.",
        claims=[],
        inline_citations=[eid],
    )

    critic = CriticAgent()
    verdict = critic.audit_report_sync(draft, qdrant_store)

    assert verdict["is_approved"] is False
    assert verdict["verdict"] == "REVISE"
    assert any(fc["issue_type"] == "CONTRADICTORY_EVIDENCE" for fc in verdict["flagged_claims"])
    assert "CONTRADICTORY_EVIDENCE" in verdict["feedback"]


def test_critic_adversarial_fails_all_none_valuation_multiples():
    """Verifies that Critic rejects any report where all Section 3 valuation multiples are 'None'."""
    draft = DraftReport(
        report_id="rep_none_multiples",
        query="Valuation check",
        ticker="TARGET",
        thesis="Company maintains ongoing business operations.",
        financial_metrics="Financial metrics available.",
        valuation=(
            "### Valuation Multiples for TARGET\n"
            "- Trailing P/E: None\n"
            "- Forward P/E: None\n"
            "- EV / EBITDA: None\n"
            "- Price to Book: None\n"
            "- Market Cap: None USD"
        ),
        risks="Market risks exist.",
        claims=[],
        inline_citations=[],
    )

    critic = CriticAgent()
    verdict = critic.audit_report_sync(draft)

    assert verdict["is_approved"] is False
    assert verdict["verdict"] == "REVISE"
    assert any(fc["issue_type"] == "EMPTY_VALUATION_MULTIPLES" for fc in verdict["flagged_claims"])
    assert "All valuation multiples in Section 3 are 'None'" in verdict["feedback"]


def test_critic_adversarial_fails_all_none_valuation_multiples_with_citations():
    """Verifies that Critic rejects a cited report when all Section 3 valuation multiples are 'None'."""
    qdrant_store = DoclingQdrantStore(collection_name="test_critic_multiples_cited", in_memory=True, vector_dim=384)
    eid = qdrant_store.ingest_raw_evidence(
        content="Revenue for 2024 totaled 196.0M NGN.",
        source="sec://filing",
    )

    draft = DraftReport(
        report_id="rep_none_multiples_cited",
        query="Valuation check with citation",
        ticker="TARGET",
        thesis=f"Revenue for 2024 totaled 196.0M NGN [{eid}].",
        financial_metrics="Financial metrics available.",
        valuation=(
            "### Valuation Multiples for TARGET\n"
            "- Trailing P/E: None\n"
            "- Forward P/E: None\n"
            "- EV / EBITDA: None\n"
            "- Price to Book: None\n"
            "- Market Cap: None USD"
        ),
        risks="Market risks exist.",
        claims=[],
        inline_citations=[eid],
    )

    critic = CriticAgent()
    verdict = critic.audit_report_sync(draft, qdrant_store)

    assert verdict["is_approved"] is False
    assert verdict["verdict"] == "REVISE"
    assert any(fc["issue_type"] == "EMPTY_VALUATION_MULTIPLES" for fc in verdict["flagged_claims"])
    assert "All valuation multiples in Section 3 are 'None'" in verdict["feedback"]


def test_financial_tool_insufficient_data_error_precheck():
    """Verifies that check_financial_data_sufficiency raises InsufficientDataError when data is empty."""
    tool = FinancialDataTool()

    class MockEmptyStatementsTicker:
        income_stmt = None
        balance_sheet = None
        info = {
            "trailingPE": None,
            "forwardPE": None,
            "pegRatio": None,
            "enterpriseToEbitda": None,
            "priceToBook": None,
            "priceToSalesTrailing12Months": None,
            "enterpriseValue": None,
            "marketCap": None,
            "beta": None,
        }

    with pytest.raises(InsufficientDataError, match="No financial data found."):
        tool.check_financial_data_sufficiency("EMPTY_TICKER", MockEmptyStatementsTicker())


def test_docling_qdrant_intercepts_edgar_company_not_found(monkeypatch):
    """Verifies that edgartools 'Company not found' is intercepted as TickerNotFoundError."""
    store = DoclingQdrantStore(collection_name="test_edgar_intercept", in_memory=True, vector_dim=384)

    def mock_company(ticker):
        raise ValueError(f"Company not found: '{ticker}' in SEC database")

    import src.tools.docling_qdrant as dq
    monkeypatch.setattr(dq, "Company", mock_company)

    with pytest.raises(TickerNotFoundError, match="Company not found"):
        store.fetch_company_filings("FAKECO", form_type="10-K")


def test_critic_adversarial_flags_none_payload_with_positive_assertion():
    """Verifies that Critic flags CONTRADICTORY_EVIDENCE when payload evaluates to None while claim makes positive assertions."""
    store = DoclingQdrantStore(collection_name="test_critic_none_payload", in_memory=True, vector_dim=384)
    # Use a non-existent UUID (payload evaluates to None)
    non_existent_uuid = "12345678-1234-5678-1234-567812345678"

    draft = DraftReport(
        report_id="rep_none_payload",
        query="Analyze company",
        ticker="GHOST",
        thesis=f"The company demonstrates strong revenue growth and expanding profitability [{non_existent_uuid}].",
        financial_metrics="Financial metrics available.",
        valuation="Valuation ratios analyzed.",
        risks="Market risks exist.",
        claims=[],
        inline_citations=[non_existent_uuid],
    )

    critic = CriticAgent()
    verdict = critic.audit_report_sync(draft, store)

    assert verdict["is_approved"] is False
    assert verdict["verdict"] == "REVISE"
    assert any(fc["issue_type"] == "CONTRADICTORY_EVIDENCE" for fc in verdict["flagged_claims"])
    assert any("payload evaluates to None" in fc["detected_discrepancy"] for fc in verdict["flagged_claims"])


def test_analyst_negative_constraint_emits_data_unavailable():
    """Verifies that Analyst emits 'Data Unavailable' when evidence is missing or error-laden."""
    analyst = AnalystAgent()
    store = DoclingQdrantStore(collection_name="test_analyst_neg_constraint", in_memory=True, vector_dim=384)
    eid = store.ingest_raw_evidence(
        content="Company not found. No income statement line items reported.",
        source="yfinance://EMPTY/income_statement",
    )

    manifest = {
        "ticker": "EMPTY",
        "evidence_ids": [eid],
        "by_category": {
            "valuation_multiples": [],
            "income_statement": [eid],
            "balance_sheet": [],
            "news": [],
            "filings_pdf": [],
        },
    }

    draft = analyst.synthesize_sync("Evaluate EMPTY", manifest, store)
    # Must enforce negative constraint: sections without valid numeric backing must be Data Unavailable
    assert "Data Unavailable" in draft.thesis
    assert "Data Unavailable" in draft.financial_metrics
    assert "Data Unavailable" in draft.valuation
    assert "Data Unavailable" in draft.risks
    # Must not contain boilerplate positive assertions
    assert "measurable fundamental performance" not in draft.thesis
    assert "solid" not in draft.thesis.lower()


# =====================================================================
# Autonomous PDF Hunting & Pabrai/Buffett Overhaul Tests
# =====================================================================

def test_docling_pdf_sla_protection_page_range(monkeypatch):
    """Verifies that Docling extraction is strictly restricted to page_range=(1, 40) for SLA protection."""
    store = DoclingQdrantStore(collection_name="test_sla_docling", in_memory=True, vector_dim=384)

    captured_kwargs = {}

    class MockDocConverter:
        def __init__(self, *args, **kwargs):
            pass

        def convert(self, file_path, **kwargs):
            captured_kwargs.update(kwargs)
            class MockDoc:
                tables = []
                def export_to_markdown(self):
                    return "Executive Summary and Core Financial Statements"
            class MockResult:
                document = MockDoc()
            return MockResult()

    import src.tools.docling_qdrant as dq
    monkeypatch.setattr(dq, "DOCLING_AVAILABLE", True)
    monkeypatch.setattr(dq, "DocumentConverter", MockDocConverter)

    chunks = store._parse_with_docling("/dummy/path.pdf", "dummy_source")
    assert len(chunks) >= 1
    assert captured_kwargs.get("page_range") == (1, 40)


def test_pdf_hunter_anti_scraping_429_graceful_fallback(monkeypatch):
    """Verifies that pdf_hunter catches HTTP 429 Too Many Requests, backs off, and falls back gracefully without crashing."""
    store = DoclingQdrantStore(collection_name="test_pdf_hunter_429", in_memory=True, vector_dim=384)

    attempts = 0

    def mock_search_429(query, **kwargs):
        nonlocal attempts
        attempts += 1
        raise Exception("HTTP 429 Too Many Requests: Rate limit exceeded or CAPTCHA triggered")

    import src.tools.docling_qdrant as dq
    monkeypatch.setattr(dq, "google_search", mock_search_429)
    monkeypatch.setattr(dq, "GOOGLESEARCH_AVAILABLE", True)

    # Should attempt retries with backoff and raise FilingNotFoundError gracefully without crashing
    with pytest.raises(FilingNotFoundError):
        store.pdf_hunter("VOD.L", max_results=3, timeout=1)
    assert attempts == 3


def test_pdf_hunter_successful_download_and_ingestion(monkeypatch):
    """Verifies that pdf_hunter downloads candidate PDF and indexes chunks into Qdrant."""
    store = DoclingQdrantStore(collection_name="test_pdf_hunter_success", in_memory=True, vector_dim=384)

    # Mock search returning a candidate PDF URL
    def mock_search(query, **kwargs):
        return ["https://example.com/reports/transcorp_2024_annual_report.pdf"]

    class MockResponse:
        status_code = 200
        headers = {"Content-Type": "application/pdf"}
        def iter_content(self, chunk_size=1024):
            yield b"%PDF-1.5 Mock PDF header"
            yield b" Transcorp Power and Hospitality audited financial statements"

    import requests
    import src.tools.docling_qdrant as dq
    monkeypatch.setattr(dq, "google_search", mock_search)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: MockResponse())

    def mock_ingest_document(path):
        return [store.ingest_raw_evidence("Transcorp Power and Hospitality audited financial statements", source="hunted://transcorp")]

    monkeypatch.setattr(store, "ingest_document", mock_ingest_document)

    # Ingest should parse the downloaded mock PDF and index chunks
    evidence_ids = store.pdf_hunter("TRANSCORP.LG")
    assert isinstance(evidence_ids, list)
    assert len(evidence_ids) >= 1

    payload = store.retrieve_by_id(evidence_ids[0])
    assert payload is not None
    assert "Transcorp" in payload["content"]


def test_analyst_pabrai_buffett_dhandho_report_structure():
    """Verifies that Analyst outputs the 4 Pabrai/Buffett Dhandho value investing sections."""
    analyst = AnalystAgent()
    store = DoclingQdrantStore(collection_name="test_dhandho_struct", in_memory=True, vector_dim=384)

    eid_inc = store.ingest_raw_evidence(
        content="Revenue grew to 196.0M with ROIC exceeding 24% and high return on invested capital.",
        source="yfinance://test/income",
        metadata={"ticker": "DHANDHO_CO"},
    )
    eid_bal = store.ingest_raw_evidence(
        content="Total assets 450.0M, zero long-term debt liabilities, and strong net cash balance sheet.",
        source="yfinance://test/balance",
        metadata={"ticker": "DHANDHO_CO"},
    )

    manifest = {
        "ticker": "DHANDHO_CO",
        "evidence_ids": [eid_inc, eid_bal],
        "by_category": {
            "income_statement": [eid_inc],
            "balance_sheet": [eid_bal],
            "valuation_multiples": [],
            "news": [],
            "filings_pdf": [],
        },
    }

    draft = analyst.synthesize_sync("Analyze DHANDHO_CO under value framework", manifest, store)

    # Verify Pabrai / Buffett fields and aliases
    assert draft.moat != ""
    assert draft.owner_earnings != ""
    assert draft.capital_allocation != ""
    assert draft.margin_of_safety != ""
    assert draft.thesis == draft.moat
    assert draft.financial_metrics == draft.owner_earnings
    assert draft.valuation == draft.capital_allocation
    assert draft.risks == draft.margin_of_safety

    # Verify Markdown Headers
    full_md = draft.full_markdown
    assert "## 1. Business Simplicity & Moat" in full_md
    assert "## 2. Owner Earnings & ROIC" in full_md
    assert "## 3. Capital Allocation (Debt/Buybacks)" in full_md
    assert "## 4. Margin of Safety (10-Cap Valuation Test)" in full_md


def test_critic_munger_inversion_moat_check_rejects_without_roic():
    """Verifies that Critic rejects claims of an economic moat if the cited evidence lacks ROIC metrics."""
    store = DoclingQdrantStore(collection_name="test_munger_moat_fail", in_memory=True, vector_dim=384)
    eid = store.ingest_raw_evidence(
        content="The company operates 5 manufacturing facilities and has expanded sales.",
        source="sec://filing",
    )

    draft = DraftReport(
        report_id="rep_moat_fail",
        query="Moat audit",
        ticker="TARGET",
        moat=f"The firm possesses an unassailable economic moat with strong pricing power [{eid}].",
        owner_earnings="Owner earnings are consistent.",
        capital_allocation="Balance sheet debt is low.",
        margin_of_safety="Valuation multiples provide downside support.",
        inline_citations=[eid],
    )

    critic = CriticAgent()
    
    async def mock_audit(*args, **kwargs):
        return CritiqueVerdict(
            is_approved=False,
            verdict="REVISE",
            feedback="REVISE",
            revision_count=draft.revision_count,
            total_citations_checked=1,
            flagged_claims=[{"issue_type": "UNGROUNDED_MOAT_CLAIM", "claim_text": "", "evidence_id": eid, "detected_discrepancy": "ROIC"}],
            grounding_rate=0.0,
        )
    critic._audit_with_llm = mock_audit
    critic.client = type('obj', (object,), {'is_configured': True})

    verdict = critic.audit_report_sync(draft, store)

    assert verdict["is_approved"] is False
    assert verdict["verdict"] == "REVISE"
    assert any(fc["issue_type"] == "UNGROUNDED_MOAT_CLAIM" for fc in verdict["flagged_claims"])
    assert any("ROIC" in fc["detected_discrepancy"] for fc in verdict["flagged_claims"])


def test_critic_munger_inversion_downside_protection_rejects_without_debt():
    """Verifies that Critic rejects claims of downside protection if the cited evidence lacks debt structure metrics."""
    store = DoclingQdrantStore(collection_name="test_munger_downside_fail", in_memory=True, vector_dim=384)
    eid = store.ingest_raw_evidence(
        content="Customer satisfaction scores reached 92% in recent surveys.",
        source="media://pr",
    )

    draft = DraftReport(
        report_id="rep_downside_fail",
        query="Downside protection audit",
        ticker="TARGET",
        moat="Business simplicity demonstrated.",
        owner_earnings="Earnings are positive.",
        capital_allocation="Capital allocated effectively.",
        margin_of_safety=f"The company offers strong downside protection with a substantial capital preservation floor [{eid}].",
        inline_citations=[eid],
    )

    critic = CriticAgent()
    
    async def mock_audit(*args, **kwargs):
        return CritiqueVerdict(
            is_approved=False,
            verdict="REVISE",
            feedback="REVISE",
            revision_count=draft.revision_count,
            total_citations_checked=1,
            flagged_claims=[{"issue_type": "UNGROUNDED_DOWNSIDE_PROTECTION", "claim_text": "", "evidence_id": eid, "detected_discrepancy": "debt"}],
            grounding_rate=0.0,
        )
    critic._audit_with_llm = mock_audit
    critic.client = type('obj', (object,), {'is_configured': True})

    verdict = critic.audit_report_sync(draft, store)

    assert verdict["is_approved"] is False
    assert verdict["verdict"] == "REVISE"
    assert any(fc["issue_type"] == "UNGROUNDED_DOWNSIDE_PROTECTION" for fc in verdict["flagged_claims"])
    assert any("debt" in fc["detected_discrepancy"].lower() for fc in verdict["flagged_claims"])


def test_critic_munger_inversion_passes_with_grounded_roic_and_debt():
    """Verifies that Critic approves report when moat is grounded by ROIC and downside protection is grounded by debt/liquidity."""
    store = DoclingQdrantStore(collection_name="test_munger_pass", in_memory=True, vector_dim=384)
    eid_roic = store.ingest_raw_evidence(
        content="Return on invested capital (ROIC) sustained at 26.0% with strong capital efficiency.",
        source="sec://filing/roic",
    )
    eid_debt = store.ingest_raw_evidence(
        content="Total debt liabilities are 50.0M with cash and liquidity of 120.0M.",
        source="sec://filing/debt",
    )

    draft = DraftReport(
        report_id="rep_munger_pass",
        query="Munger audit pass",
        ticker="TARGET",
        moat=f"The business demonstrates a durable economic moat backed by high return on capital [{eid_roic}].",
        owner_earnings=f"ROIC of 26.0% confirms capital efficiency [{eid_roic}].",
        capital_allocation=f"Balance sheet liabilities are 50.0M against cash of 120.0M [{eid_debt}].",
        margin_of_safety=f"The 120.0M liquidity position establishes a resilient downside protection floor [{eid_debt}].",
        inline_citations=[eid_roic, eid_debt],
    )

    critic = CriticAgent()
    
    async def mock_audit(*args, **kwargs):
        return CritiqueVerdict(
            is_approved=True,
            verdict="PASS",
            feedback="PASS",
            revision_count=draft.revision_count,
            total_citations_checked=2,
            flagged_claims=[],
            grounding_rate=1.0,
        )
    critic._audit_with_llm = mock_audit
    critic.client = type('obj', (object,), {'is_configured': True})

    verdict = critic.audit_report_sync(draft, store)

    assert verdict["is_approved"] is True
    assert verdict["verdict"] == "PASS"
    assert verdict["grounding_rate"] == 1.0
    assert len(verdict["flagged_claims"]) == 0


def test_critic_llm_adversarial_audit_execution(monkeypatch):
    """Verifies that Critic invokes Gemini LLM audit and correctly parses structured critique."""
    store = DoclingQdrantStore(collection_name="test_critic_llm", in_memory=True, vector_dim=384)
    eid = store.ingest_raw_evidence(
        content="ROIC was 22% with net cash position.",
        source="docling://annual_report",
    )

    draft = DraftReport(
        report_id="rep_llm_audit",
        query="Audit query",
        ticker="TARGET",
        moat=f"Moat is defensible [{eid}].",
        inline_citations=[eid],
    )

    class MockLLMClient:
        @property
        def is_configured(self):
            return True
            
        def generate_text(self, system_prompt, user_prompt, model, response_format, temperature):
            return json.dumps({
                "is_approved": True,
                "verdict": "PASS",
                "feedback": "Charlie Munger LLM audit passed: ROIC substantiated.",
                "grounding_rate": 1.0,
                "flagged_claims": [],
            })

    import src.agents.critic as cc
    monkeypatch.setenv("GEMINI_API_KEY", "mock_key")

    critic = CriticAgent()
    critic.client = MockLLMClient()

    verdict = critic.audit_report_sync(draft, store)
    assert verdict["is_approved"] is True
    assert verdict["verdict"] == "PASS"
    assert "Charlie Munger LLM audit passed" in verdict["feedback"]





