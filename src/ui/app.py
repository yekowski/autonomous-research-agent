"""Streamlit Dashboard for Autonomous Research Agent.

Visualizes the 4-node adversarial research pipeline and LLMOps metrics in real time:
- Planner -> Researcher -> Analyst -> Critic (Adversarial Audit Loop)
- SLA Latency tracking & 100% Grounding Rate Quality Gate
- Point-in-time Qdrant Evidence Manifest inspection
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

import streamlit as st

from src.eval.llmops_harness import EvalRunRecord, EvaluationHarness

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# Page Configuration & Institutional Theme
# =====================================================================

st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Institutional Look & Feel
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
    }
    .badge-pass {
        background-color: #dcfce7;
        color: #15803d;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-fail {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .citation-tag {
        background-color: #e0f2fe;
        color: #0369a1;
        font-family: monospace;
        padding: 0.15rem 0.4rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Sidebar: System Config & Overview
# =====================================================================

with st.sidebar:
    st.markdown("### 🏛️ System Architecture")
    st.markdown(
        """
        **Closed-World Evidence Multi-Agent Pipeline**:
        1. **Planner**: Cognitive task DAG decomposition
        2. **Researcher**: Scraping & Qdrant vector indexing
        3. **Analyst**: Thesis & valuation with point-in-time citations
        4. **Critic (Adversarial)**: Forensic audit & 3-loop revision gate
        """
    )
    st.divider()
    st.markdown("### ⏱️ SLA & Quality Targets")
    
    # Define dynamic SLA based on LLM Provider
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    max_sla = 1200.0 if provider == "ollama" else 90.0
    default_sla = 600.0 if provider == "ollama" else 45.0
    
    st.markdown(f"- **Latency SLA**: $\\le$ {default_sla}s ceiling")
    st.markdown("- **Grounding Quality Gate**: 100% strictly verified")
    st.markdown("- **Max Revision Cycles**: 3 loops cutoff")
    
    sla_input = st.slider("SLA Timeout (seconds)", min_value=15.0, max_value=max_sla, value=default_sla, step=5.0)
    pdf_url = st.sidebar.text_input("Direct Annual Report PDF URL (Optional)")
    st.divider()
    st.markdown("### 🔑 LLM API Configuration")
    
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["Gemini", "Ollama"],
        index=0,
        help="Select between Gemini (API) or Ollama (Local) for inference."
    )
    os.environ["LLM_PROVIDER"] = llm_provider.lower()
    
    if llm_provider == "Ollama":
        ollama_model_input = st.selectbox(
            "Ollama Model Name",
            options=["llama3.1", "qwen3:14b", "llama3.2:3b", "llama3", "mistral"],
            index=0,
            help="Select the model pulled in your local Ollama instance."
        )
        os.environ["OLLAMA_MODEL"] = ollama_model_input.strip()
        st.caption(f"🦙 Ollama Local Active ({os.environ['OLLAMA_MODEL']})")
    else:
        existing_key = os.environ.get("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            "Gemini API Key (Optional)",
            value=existing_key,
            type="password",
            help="Provide your GEMINI_API_KEY for dynamic synthesis with Gemini 3.1 Pro.",
        )
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input.strip()
            st.caption("🟢 Gemini LLM Active (gemini-3.1-pro-preview)")
        else:
            st.caption("🟡 Deterministic Mode Active (Offline / Test Mode)")

# =====================================================================
# Header & Query Input
# =====================================================================

st.markdown('<div class="main-header">Autonomous Research Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Institutional Equity Research Pipeline with Adversarial Forensic Auditing & Qdrant Grounding</div>',
    unsafe_allow_html=True,
)

col_input, col_action = st.columns([4, 1], gap="medium")

with col_input:
    target_company = st.text_input(
        "Target Company / Ticker",
        value="Transcorp Group",
        placeholder="e.g. Transcorp Group, Beta Glass Plc, AAPL",
        help="Specify the target company or stock ticker symbol to analyze.",
    )

with col_action:
    st.write("")  # Spacing
    st.write("")
    run_button = st.button("🚀 Run Research", type="primary", use_container_width=True)

# Quick Benchmark Presets
st.markdown(
    "<small style='color: #64748b;'>Benchmark Presets: "
    "<b>Transcorp Group</b> | <b>Beta Glass Plc</b> | <b>AAPL</b>"
    "</small>",
    unsafe_allow_html=True,
)

st.divider()

# Session State Persistence
if "last_run_record" not in st.session_state:
    st.session_state["last_run_record"] = None

# =====================================================================
# Async Execution Wrapper
# =====================================================================

if run_button:
    if not target_company.strip():
        st.warning("Please enter a valid target company name or ticker symbol.")
    else:
        # Status Containers for Real-Time Execution Tracking
        with st.status(f"Executing Research Pipeline for '{target_company}'...", expanded=True) as status_box:
            step_messages = []

            def on_node_progress(msg: str) -> None:
                step_messages.append(msg)
                status_box.write(f"• {msg}")

            try:
                # Initialize Evaluation Harness with configured SLA
                harness = EvaluationHarness(sla_seconds=sla_input)

                # Execute pipeline using asyncio.run()
                record: EvalRunRecord = asyncio.run(
                    harness.run_pipeline_for_target(
                        target_company=target_company.strip(),
                        progress_callback=on_node_progress,
                        pdf_url=pdf_url.strip() if pdf_url else None,
                    )
                )

                st.session_state["last_run_record"] = record

                if record.status == "PASS":
                    status_box.update(
                        label=f"Pipeline Completed: PASS (Latency: {record.latencies.total_e2e_latency_s:.2f}s, Loops: {record.adversarial_loops_triggered})",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status_box.update(
                        label=f"Pipeline Completed: {record.status} (Grounding: {record.grounding_rate:.0%}, Loops: {record.adversarial_loops_triggered})",
                        state="error" if record.status in ("FAIL", "SLA_VIOLATION") else "complete",
                        expanded=True,
                    )

            except Exception as exc:
                status_box.update(label=f"Pipeline execution failed: {exc}", state="error")
                st.error(f"Error during execution: {exc}")
                logger.exception("Streamlit pipeline execution failure")

# =====================================================================
# Metrics Dash (EvalRunRecord)
# =====================================================================

record: Optional[EvalRunRecord] = st.session_state.get("last_run_record")

if record is not None:
    st.subheader("📊 LLMOps Performance & Quality Metrics")

    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

    with m_col1:
        sla_met = record.latencies.total_e2e_latency_s <= sla_input
        st.metric(
            label="Total Latency",
            value=f"{record.latencies.total_e2e_latency_s:.2f}s",
            delta=f"SLA {'≤' if sla_met else '>'} {sla_input:.0f}s ({'Pass' if sla_met else 'Breach'})",
            delta_color="normal" if sla_met else "inverse",
        )

    with m_col2:
        grounding_met = record.grounding_rate >= 1.0
        st.metric(
            label="Grounding Rate",
            value=f"{record.grounding_rate:.1%}",
            delta="100% Quality Gate" if grounding_met else "Un-cited claims flagged",
            delta_color="normal" if grounding_met else "inverse",
        )

    with m_col3:
        st.metric(
            label="Critic Revision Loops",
            value=f"{record.adversarial_loops_triggered} / 3",
            delta="Approved by Critic" if record.is_approved_by_critic else "Rejected / Max Capped",
            delta_color="normal" if record.is_approved_by_critic else "inverse",
        )

    with m_col4:
        st.metric(
            label="Audit Verdict",
            value=record.status,
            delta="Audited & Grounded" if record.status == "PASS" else "Quality/SLA Issue",
            delta_color="normal" if record.status == "PASS" else "inverse",
        )

    with m_col5:
        st.metric(
            label="Planner / Critic Latency",
            value=f"{record.latencies.planner_latency_ms:.0f}ms / {record.latencies.critic_latency_ms:.0f}ms",
            delta=f"Res: {record.latencies.researcher_latency_ms:.0f}ms | Ana: {record.latencies.analyst_latency_ms:.0f}ms",
            delta_color="off",
        )

    if record.failure_reasons:
        with st.expander("⚠️ Audit Findings & Quality Gate Warnings", expanded=True):
            for reason in record.failure_reasons:
                st.error(f"• {reason}")

    st.divider()

    # =====================================================================
    # Output Tabs: Pabrai/Buffett Framework & Evidence Store
    # =====================================================================

    # Defensive getters to support both new DraftReport objects, legacy session state objects, and dicts
    def _extract_report_field(report: Any, *field_names: str) -> str:
        if report is None:
            return ""
        if isinstance(report, dict):
            for name in field_names:
                val = report.get(name)
                if val:
                    return str(val)
            return ""
        for name in field_names:
            val = getattr(report, name, None)
            if val:
                return str(val)
        return ""

    class AmbiguousCitationError(Exception): pass

    def _extract_citations(report: Any, manifest: Optional[Dict[str, Any]] = None) -> List[str]:
        if report is None:
            return []
        
        raw_citations = []
        if isinstance(report, dict):
            raw_citations = list(report.get("inline_citations") or [])
        else:
            raw_citations = list(getattr(report, "inline_citations", []) or [])
            
        if not manifest:
            return raw_citations
            
        manifest_uuids = set()
        if "indexed_items" in manifest:
            for item in manifest["indexed_items"]:
                if "evidence_id" in item:
                    manifest_uuids.add(item["evidence_id"])
        elif "evidence_ids" in manifest:
            manifest_uuids.update(manifest["evidence_ids"])
        elif isinstance(manifest, dict):
            manifest_uuids.update(manifest.keys())
            
        citations = []
        for raw_id in raw_citations:
            found = [uid for uid in manifest_uuids if uid.startswith(raw_id)]
            if len(found) == 1:
                citations.append(found[0])
            elif len(found) > 1:
                raise AmbiguousCitationError(f"Citation [{raw_id}] matches multiple items in the manifest.")
            else:
                citations.append(raw_id)
                
        return list(set(citations))

    report_citations = _extract_citations(record.draft_report, record.evidence_manifest)
    moat_content = _extract_report_field(record.draft_report, "moat", "thesis")
    earnings_content = _extract_report_field(record.draft_report, "owner_earnings", "financial_metrics")
    capital_content = _extract_report_field(record.draft_report, "capital_allocation", "valuation")
    margin_content = _extract_report_field(record.draft_report, "margin_of_safety", "risks")

    if record.draft_report:
        st.markdown(f"### Dhandho Value Investment Analysis: {record.target_company}")

        # Badges
        status_badge = (
            '<span class="badge-pass">PASS: 100% Grounded</span>'
            if record.status == "PASS"
            else f'<span class="badge-fail">{record.status}</span>'
        )
        st.markdown(
            f"**Audit Status**: {status_badge} &nbsp;|&nbsp; "
            f"**Inline Citations**: `{len(report_citations)} references`",
            unsafe_allow_html=True,
        )
        st.markdown("---")

    tab_moat, tab_earnings, tab_capital, tab_margin, tab_evidence = st.tabs(
        ["🏰 Moat", "💰 Owner Earnings", "⚖️ Capital Allocation", "🛡️ Margin of Safety", "🗄️ Evidence Store"]
    )

    with tab_moat:
        st.markdown("### 1. Business Simplicity & Moat")
        st.caption("Circle of competence, durable competitive advantage, pricing power, and business model simplicity.")
        if record.draft_report:
            st.markdown(moat_content or "No moat analysis available.")
        else:
            st.info("No report generated for this target execution.")

    with tab_earnings:
        st.markdown("### 2. Owner Earnings & ROIC")
        st.caption("Operating cash flow minus maintenance capex, return on invested capital (ROIC) vs cost of capital.")
        if record.draft_report:
            st.markdown(earnings_content or "No owner earnings analysis available.")
        else:
            st.info("No report generated for this target execution.")

    with tab_capital:
        st.markdown("### 3. Capital Allocation (Debt/Buybacks)")
        st.caption("Balance sheet leverage, debt commitments, share buybacks, and management discipline.")
        if record.draft_report:
            st.markdown(capital_content or "No capital allocation analysis available.")
        else:
            st.info("No report generated for this target execution.")

    with tab_margin:
        st.markdown("### 4. Margin of Safety (10-Cap Valuation Test)")
        st.caption("10-cap pre-tax earnings yield test, downside capital preservation floor, and asymmetric risk/reward.")
        if record.draft_report:
            st.markdown(margin_content or "No margin of safety analysis available.")
            if report_citations:
                with st.expander(f"📌 All Inline Citations ({len(report_citations)})"):
                    for cid in report_citations:
                        st.markdown(f"- <span class='citation-tag'>[{cid}]</span>", unsafe_allow_html=True)
        else:
            st.info("No report generated for this target execution.")

    with tab_evidence:
        st.markdown("### Qdrant Vector Evidence Store & Ingestion Manifest")
        st.caption(
            "Every claim made by the Analyst must trace directly back to an evidence record indexed by the Researcher."
        )

        manifest = record.evidence_manifest or {}

        if manifest:
            # Build normalized items mapping: supports both EvidenceManifest dict and eid->payload dict
            evidence_items: Dict[str, Any] = {}
            if isinstance(manifest, dict) and "evidence_ids" in manifest and isinstance(manifest["evidence_ids"], list):
                total_count = manifest.get("total_indexed", len(manifest["evidence_ids"]))
                for eid in manifest["evidence_ids"]:
                    payload = None
                    if record.qdrant_store:
                        payload = record.qdrant_store.retrieve_by_id(eid)
                    evidence_items[eid] = payload or {
                        "evidence_id": eid,
                        "source": f"Indexed Qdrant Chunk ({eid[:8]})",
                        "content": f"Evidence record {eid}",
                    }
            elif isinstance(manifest, dict):
                total_count = len(manifest)
                evidence_items = manifest
            elif isinstance(manifest, list):
                total_count = len(manifest)
                evidence_items = {
                    (item if isinstance(item, str) else str(getattr(item, "evidence_id", item))): item
                    for item in manifest
                }
            else:
                total_count = 0
                evidence_items = {}

            m_summary_col1, m_summary_col2 = st.columns(2)
            with m_summary_col1:
                st.info(f"**Total Indexed Evidence Items**: {total_count}")
            with m_summary_col2:
                cited_count = len(report_citations)
                st.success(f"**Citations Utilized in Synthesis**: {cited_count}")

            # Structured view of each Evidence Chunk
            st.markdown("#### Indexed Evidence Manifest Items")
            for eid, meta in evidence_items.items():
                # Defensively normalize meta to a dict if it is a str or serialized JSON
                if isinstance(meta, dict):
                    meta_dict = meta
                elif isinstance(meta, str):
                    try:
                        parsed = json.loads(meta)
                        meta_dict = parsed if isinstance(parsed, dict) else {"content": meta, "source": meta}
                    except Exception:
                        meta_dict = {"content": meta, "source": meta}
                else:
                    meta_dict = {"content": str(meta), "source": "Unknown Source"}

                # Safely extract sub-metadata if nested under 'metadata' key
                meta_nested = meta_dict.get("metadata") if isinstance(meta_dict, dict) else None
                if isinstance(meta_nested, dict):
                    ticker = meta_dict.get("ticker") or meta_nested.get("ticker")
                    task_action = meta_dict.get("task_action") or meta_nested.get("task_action") or meta_nested.get("category")
                    timestamp = meta_dict.get("timestamp") or meta_nested.get("timestamp")
                    chunk_type = meta_dict.get("chunk_type") or meta_nested.get("chunk_type")
                else:
                    ticker = meta_dict.get("ticker") if isinstance(meta_dict, dict) else None
                    task_action = meta_dict.get("task_action") if isinstance(meta_dict, dict) else None
                    timestamp = meta_dict.get("timestamp") if isinstance(meta_dict, dict) else None
                    chunk_type = meta_dict.get("chunk_type") if isinstance(meta_dict, dict) else None

                is_cited = eid in report_citations
                title_prefix = "✅ [CITED]" if is_cited else "📄 [INDEXED]"
                source = str((meta_dict.get("source") if isinstance(meta_dict, dict) else None) or "Unknown Source")
                topic = str((meta_dict.get("topic") if isinstance(meta_dict, dict) else None) or task_action or chunk_type or "Evidence")

                with st.expander(f"{title_prefix} Evidence ID: `{eid}` — {source} ({topic})"):
                    col_m1, col_m2 = st.columns([1, 2])
                    with col_m1:
                        st.write("**Metadata**:")
                        st.json(
                            {
                                "evidence_id": eid,
                                "source": source,
                                "timestamp": timestamp,
                                "ticker": ticker,
                                "task_action": task_action,
                                "chunk_type": chunk_type,
                            }
                        )
                    with col_m2:
                        st.write("**Extracted Content Summary**:")
                        excerpt = (meta_dict.get("summary") or meta_dict.get("text") or meta_dict.get("content")) if isinstance(meta_dict, dict) else str(meta_dict)
                        st.markdown(str(excerpt or "No excerpt available."))

            # Raw JSON View
            st.markdown("#### Raw Manifest JSON")
            st.json(manifest)
        else:
            st.warning("No evidence manifest recorded in this run.")

else:
    st.info("👋 Enter a target company above and click **Run Research** to launch the 4-node autonomous research pipeline.")
