"""Planner Agent Node.

Responsible for decomposing high-level user research inquiries into a strict,
structured JSON list of discrete research tasks mapped to available tools.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)


# =====================================================================
# Available Tool Actions & Task Schemas
# =====================================================================

class AllowedToolAction(str, Enum):
    PULL_INCOME_STATEMENT = "pull_income_statement"
    PULL_BALANCE_SHEET = "pull_balance_sheet"
    PULL_VALUATION_MULTIPLES = "pull_valuation_multiples"
    FETCH_RECENT_NEWS = "fetch_recent_news"
    FETCH_COMPANY_FILINGS = "fetch_company_filings"
    FETCH_SEC_EDGAR_FILINGS = "fetch_company_filings"  # Backward-compatible alias
    INGEST_LATEST_10K_PDF = "fetch_company_filings"  # Backward-compatible alias


@dataclass
class ResearchTask:
    """Individual decomposed sub-task strictly mapped to a registered tool action."""
    task_id: str
    action: str  # Must be one of AllowedToolAction
    ticker: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "action": self.action,
            "ticker": self.ticker,
            "parameters": self.parameters,
            "rationale": self.rationale,
        }


@dataclass
class ResearchPlan:
    """Executable research plan holding the list of planned tasks."""
    plan_id: str
    objective: str
    ticker: str
    tasks: List[ResearchTask] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json_list(self) -> List[Dict[str, Any]]:
        return [t.to_dict() for t in self.tasks]


# =====================================================================
# Production Research Planner
# =====================================================================

class ResearchPlanner:
    """Lead Research Strategist.
    
    Decomposes user financial inquiries into a strict JSON list of research tasks
    mapped to: pull_income_statement, pull_balance_sheet, pull_valuation_multiples,
    fetch_recent_news, and fetch_company_filings.
    """

    SYSTEM_PROMPT = """You are the Lead Research Strategist for an institutional equity research team.
Your sole job is to decompose the user's financial question into a strict JSON list of actionable research tasks.

CRITICAL TICKER RESOLUTION INSTRUCTIONS:
You MUST extract the core company from the user's query and resolve it to a valid, precise Yahoo Finance ticker symbol. 
Do NOT paste the user's raw query into the ticker field. 
- For US companies, use the standard uppercase ticker (e.g., AAPL, NVDA).
- For international companies, you MUST append the correct Yahoo Finance exchange suffix. 
  - Nigerian Exchange (NGX) equities MUST have the '.LG' suffix (e.g., 'MTNN.LG' for MTN Nigeria, 'TRANSCORP.LG' for Transcorp, 'BETAGLAS.LG' for Beta Glass).
  - London Stock Exchange MUST have the '.L' suffix.
  - Toronto Stock Exchange MUST have the '.TO' suffix.

Available Tools:
- "pull_income_statement": Retrieves annual and quarterly income statements (revenue, margins, net income).
- "pull_balance_sheet": Retrieves balance sheet data (assets, liabilities, debt, cash).
- "pull_valuation_multiples": Retrieves P/E, forward P/E, EV/EBITDA, P/B, market cap, and valuation metrics.
- "fetch_recent_news": Retrieves latest corporate news, catalyst events, and press releases.
- "fetch_company_filings": A market-aware jurisdiction router. Automatically fetches regulatory filings (such as SEC 10-Ks for US equities via edgartools) or autonomously hunts PDF annual reports for international markets (.LG, .L, .TO), indexing extracted chunks into Qdrant. Universal tool for all corporate filing ingestion.

You must respond ONLY with a valid JSON array of task objects. Do NOT include markdown code fences or conversational text.
Each task object MUST have:
{
  "task_id": "string",
  "action": "pull_income_statement" | "pull_balance_sheet" | "pull_valuation_multiples" | "fetch_recent_news" | "fetch_company_filings",
  "ticker": "string (MUST be the resolved Yahoo Finance ticker symbol, e.g. 'BETAGLAS.LG', NEVER a full sentence)",
  "parameters": {},
  "rationale": "string"
}
"""

    def __init__(self, model_name: str = "gemini-3.1-pro-preview") -> None:
        """Initializes the Planner Agent."""
        self.model_name = model_name
        self.client = LLMClient()

    def plan_research(self, query: str) -> List[Dict[str, Any]]:
        """Takes a user financial query and produces a strict JSON list of research tasks.
        
        Args:
            query: User financial question (e.g. "Is Transcorp Group a buy?", "Analyze NVDA valuation").

        Returns:
            List[Dict[str, Any]]: Strict JSON task list mapping to available tools.
        """
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty.")

        try:
            if self.client.is_configured:
                return self._call_llm_for_plan(clean_query)
        except Exception as e:
            logger.warning("LLM call failed (%s), falling back to deterministic planner.", e)

        # Robust deterministic task generator
        return self._generate_deterministic_plan(clean_query)

    def _call_llm_for_plan(self, query: str) -> List[Dict[str, Any]]:
        """Invokes LLM with structured output."""
        raw_text = self.client.generate_text(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=query,
            model=self.model_name,
            response_format="json",
            temperature=0.1
        )
        
        try:
            tasks_data = json.loads(raw_text)
            if isinstance(tasks_data, dict) and "tasks" in tasks_data:
                tasks_data = tasks_data["tasks"]

            if isinstance(tasks_data, list):
                return self._validate_and_normalize_tasks(tasks_data, query)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response from LLM: %s", raw_text)

        return self._generate_deterministic_plan(query)

    def _generate_deterministic_plan(self, query: str) -> List[Dict[str, Any]]:
        """Rule-based plan generator extracting ticker and assigning tool actions."""
        ticker = self._extract_ticker_or_name(query)

        task_specs = [
            (
                AllowedToolAction.PULL_VALUATION_MULTIPLES.value,
                f"Obtain key valuation multiples (P/E, forward P/E, EV/EBITDA, market cap) for {ticker}.",
                {},
            ),
            (
                AllowedToolAction.PULL_INCOME_STATEMENT.value,
                f"Evaluate revenue trajectory, operating margins, and profit trends for {ticker}.",
                {"quarterly": False},
            ),
            (
                AllowedToolAction.PULL_BALANCE_SHEET.value,
                f"Assess debt structure, liquidity, and leverage ratios for {ticker}.",
                {"quarterly": False},
            ),
            (
                AllowedToolAction.FETCH_RECENT_NEWS.value,
                f"Identify near-term catalysts, regulatory disclosures, and market sentiment for {ticker}.",
                {"limit": 10},
            ),
            (
                AllowedToolAction.FETCH_COMPANY_FILINGS.value,
                f"Retrieve and index company regulatory filings (SEC 10-K for US, or local disclosures) for {ticker}.",
                {"form_type": "10-K"},
            ),
        ]

        tasks: List[Dict[str, Any]] = []
        for idx, (action, rationale, params) in enumerate(task_specs, start=1):
            tasks.append({
                "task_id": f"task_{idx:03d}",
                "action": action,
                "ticker": ticker,
                "parameters": params,
                "rationale": rationale,
            })

        return tasks

    def _extract_ticker_or_name(self, query: str) -> str:
        """Extracts plausible ticker symbol or corporate identifier from query."""
        # Check for explicit ticker in parentheses e.g. "Transcorp Group (TRANSCORP)" or "Apple (AAPL)"
        paren_match = re.search(r"\(([A-Z0-9\.\-]+)\)", query)
        if paren_match:
            return paren_match.group(1).upper()

        # Known common entity name to ticker heuristics
        q_upper = query.upper()
        if "TRANSCORP" in q_upper:
            return "TRANSCORP.LG"
        if "APPLE" in q_upper:
            return "AAPL"
        if "NVIDIA" in q_upper or "NVDA" in q_upper:
            return "NVDA"
        if "MICROSOFT" in q_upper or "MSFT" in q_upper:
            return "MSFT"
        if "TESLA" in q_upper or "TSLA" in q_upper:
            return "TSLA"

        # Look for capitalized ticker words (2 to 5 uppercase chars)
        words = query.split()
        for w in words:
            clean = re.sub(r"[^A-Za-z0-9]", "", w)
            if clean.isupper() and 2 <= len(clean) <= 5:
                return clean

        # Default to first prominent phrase or sanitized query
        cleaned = re.sub(r"(?i)^(is|what is|analyze|should i buy|buy|sell)\s+", "", query).strip()
        cleaned = re.sub(r"(?i)\s+(a buy|a sell|worth buying|good investment)\??$", "", cleaned).strip()
        return cleaned or "TARGET"

    def _validate_and_normalize_tasks(self, raw_tasks: List[Any], query: str) -> List[Dict[str, Any]]:
        """Ensures all tasks conform strictly to the allowed tool actions and schemas."""
        allowed_actions = {a.value for a in AllowedToolAction}
        normalized: List[Dict[str, Any]] = []

        default_ticker = self._extract_ticker_or_name(query)

        for idx, t in enumerate(raw_tasks, start=1):
            if not isinstance(t, dict):
                continue
            action = str(t.get("action", "")).strip().lower()
            if action not in allowed_actions:
                # Fuzzy map action if close
                if "income" in action:
                    action = AllowedToolAction.PULL_INCOME_STATEMENT.value
                elif "balance" in action:
                    action = AllowedToolAction.PULL_BALANCE_SHEET.value
                elif "multiple" in action or "valuation" in action or "pe" in action:
                    action = AllowedToolAction.PULL_VALUATION_MULTIPLES.value
                elif "news" in action:
                    action = AllowedToolAction.FETCH_RECENT_NEWS.value
                elif "filing" in action or "edgar" in action or "sec" in action or "pdf" in action or "10k" in action:
                    action = AllowedToolAction.FETCH_COMPANY_FILINGS.value
                else:
                    continue

            normalized.append({
                "task_id": str(t.get("task_id") or f"task_{idx:03d}"),
                "action": action,
                "ticker": str(t.get("ticker") or default_ticker).upper(),
                "parameters": t.get("parameters", {}) if isinstance(t.get("parameters"), dict) else {},
                "rationale": str(t.get("rationale") or f"Execute {action}"),
            })

        return normalized or self._generate_deterministic_plan(query)

    # Compatibility method with previous interface
    def generate_plan(self, objective: str, constraints: Optional[Dict[str, Any]] = None) -> ResearchPlan:
        tasks_json = self.plan_research(objective)
        ticker = tasks_json[0]["ticker"] if tasks_json else "TARGET"
        tasks = [
            ResearchTask(
                task_id=t["task_id"],
                action=t["action"],
                ticker=t["ticker"],
                parameters=t.get("parameters", {}),
                rationale=t.get("rationale", ""),
            )
            for t in tasks_json
        ]
        return ResearchPlan(
            plan_id=str(uuid.uuid4()),
            objective=objective,
            ticker=ticker,
            tasks=tasks,
            metadata={"constraints": constraints or {}},
        )


# Backward compatibility alias
PlannerAgent = ResearchPlanner
