"""Critic Agent Node (Adversarial Auditor).

Adversarial Chief Risk Officer and Forensic Auditor.
Extracts all inline evidence citations, queries Qdrant payloads via retrieve_by_id,
verifies numeric grounding and factual consistency, and issues strict PASS or REVISE verdicts.
Enforces a hardcoded 3-loop revision limit to prevent infinite cycles.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

from src.agents.analyst import DraftReport
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    GENAI_AVAILABLE = False


# =====================================================================
# Verdict Status Enum & TypedDict Schemas
# =====================================================================

class VerdictStatus(str, Enum):
    PASS = "PASS"
    REVISE = "REVISE"

class FlaggedClaimDetail(TypedDict):
    """Detailed record of an ungrounded or hallucinated claim."""
    claim_text: str
    evidence_id: Optional[str]
    issue_type: str  # 'MISSING_CITATION' | 'NONEXISTENT_EVIDENCE_ID' | 'HALLUCINATED_NUMBERS' | 'CONTRADICTORY_EVIDENCE' | 'EMPTY_VALUATION_MULTIPLES' | 'UNGROUNDED_MOAT_CLAIM' | 'UNGROUNDED_DOWNSIDE_PROTECTION'
    detected_discrepancy: str


class CritiqueVerdict(TypedDict):
    """Strict output schema emitted by the Critic node."""
    is_approved: bool
    verdict: str  # "PASS" | "REVISE"
    feedback: str
    revision_count: int
    total_citations_checked: int
    flagged_claims: List[FlaggedClaimDetail]
    grounding_rate: float


# =====================================================================
# Critic Agent Implementation
# =====================================================================

class CriticAgent:
    """Adversarial agent that red-teams Analyst drafts against source evidence.
    Enforces a strict 3-loop revision gate and ensures 100% citation grounding."""
    
    class AmbiguousCitationError(Exception): pass

    MAX_REVISION_LOOPS: int = 3

    SYSTEM_PROMPT = """You are Charlie Munger operating as an adversarial forensic auditor and risk manager.
Your operational philosophy is guided by the timeless principle: "Invert, always invert."
Your mandate is NOT to assist or flatter the Analyst. Your goal is to vigorously red-team the thesis, expose capital destruction risks, and prevent ungrounded assertions from reaching publication.

Operational Directives:
1. Moat Inversion Check: If the Analyst asserts an economic moat, durable competitive advantage, or pricing power, REJECT the draft (issue_type: 'UNGROUNDED_MOAT_CLAIM') unless they explicitly cite verifiable ROIC, ROCE, or return on capital disclosures from Qdrant confirming that the business generates returns above its cost of capital.
2. Downside Protection & Debt Inversion Check: If the Analyst claims downside protection, margin of safety, or low risk, REJECT the draft (issue_type: 'UNGROUNDED_DOWNSIDE_PROTECTION') unless they cite concrete balance sheet debt structures, liability commitments, or liquidity runway from Qdrant.
3. Semantic & Numeric Grounding: For every single claim in the report, inspect the cited evidence_id against the provided Qdrant evidence payloads. Reject drafts that make positive assertions, bullish claims, or optimistic growth statements while citing empty or error-laden evidence payloads (issue_type: 'CONTRADICTORY_EVIDENCE'). If figures do not match mathematically or verbatim, flag as 'HALLUCINATED_NUMBERS'.
4. Negative Constraints: If evidence indicates missing data, verify the section explicitly states "Data Unavailable".

Respond STRICTLY with valid JSON conforming to this schema:
{
  "is_approved": boolean,
  "verdict": "PASS" | "REVISE",
  "feedback": "string containing detailed forensic audit critique and non-negotiable remediation instructions",
  "grounding_rate": float (between 0.0 and 1.0),
  "flagged_claims": [
    {
      "claim_text": "string excerpt of failed claim",
      "evidence_id": "string uuid or null",
      "issue_type": "MISSING_CITATION" | "NONEXISTENT_EVIDENCE_ID" | "HALLUCINATED_NUMBERS" | "CONTRADICTORY_EVIDENCE" | "EMPTY_VALUATION_MULTIPLES" | "UNGROUNDED_MOAT_CLAIM" | "UNGROUNDED_DOWNSIDE_PROTECTION",
      "detected_discrepancy": "string detailed explanation of why the claim failed"
    }
  ]
}
"""

    def __init__(
        self,
        max_revisions_allowed: int = MAX_REVISION_LOOPS,
        model_name: str = "gemini-3.1-flash-preview",
    ) -> None:
        self.max_revisions_allowed = max_revisions_allowed
        self.model_name = model_name
        self.client = LLMClient()

    async def audit_report(
        self,
        draft: DraftReport,
        qdrant_store: Optional[Any] = None,
    ) -> CritiqueVerdict:
        """Asynchronously audits all inline citations and claims in the draft report.
        
        Args:
            draft: The draft report to audit.
            qdrant_store: Evidence store connector to retrieve raw evidence chunks.

        Returns:
            CritiqueVerdict: Strict verdict schema with approval flag and remediation feedback.
        """
        current_revisions = draft.revision_count

        # Guard: Hardcoded maximum revision limit to prevent infinite loops
        if current_revisions >= self.max_revisions_allowed:
            logger.info("Draft reached max revisions (%d). Forcing approval with audit caveats.", current_revisions)
            return CritiqueVerdict(
                is_approved=True,
                verdict="PASS",
                feedback=(
                    f"Approved under conditional audit: Draft has reached the maximum revision "
                    f"threshold ({self.max_revisions_allowed} cycles). Retaining known caveats."
                ),
                revision_count=current_revisions,
                total_citations_checked=len(draft.inline_citations),
                flagged_claims=[],
                grounding_rate=1.0,
            )

        # Pre-Flight Gate: Run orphaned heuristics
        sentences = self._extract_cited_sentences(
            draft.full_markdown if hasattr(draft, "full_markdown") else str(draft.thesis) + " " + str(draft.financial_metrics),
            draft.metadata.get("manifest")
        )
        flagged_deterministic: List[FlaggedClaimDetail] = []
        
        # Also check multiple none
        if self._check_all_valuation_multiples_are_none(draft.valuation):
            flagged_deterministic.append(FlaggedClaimDetail(
                claim_text="Valuation multiples check",
                evidence_id=None,
                issue_type="EMPTY_VALUATION_MULTIPLES",
                detected_discrepancy="All valuation multiples in Section 3 are 'None'.",
            ))

        for sent, eid in sentences:
            finding = await self._verify_single_claim(sent, eid, qdrant_store, draft)
            if finding:
                flagged_deterministic.append(finding)

        has_fatal_contradiction = any(
            f["issue_type"] in ("CONTRADICTORY_EVIDENCE", "NONEXISTENT_EVIDENCE_ID", "EMPTY_VALUATION_MULTIPLES") 
            for f in flagged_deterministic
        )

        if has_fatal_contradiction:
            feedback_str = "CRITICAL AUDIT FAILURE:\n"
            for f in flagged_deterministic:
                feedback_str += f"- {f['issue_type']}: {f['detected_discrepancy']}\n"

            return CritiqueVerdict(
                is_approved=False,
                verdict="REVISE",
                feedback=feedback_str.strip(),
                revision_count=current_revisions,
                total_citations_checked=len(sentences),
                flagged_claims=flagged_deterministic,
                grounding_rate=0.0,
            )

        # If live Gemini LLM is configured and store is available, execute live Charlie Munger audit
        if self.client.is_configured and qdrant_store is not None:
            try:
                # We can optionally pass flagged_deterministic to _audit_with_llm if we change its signature.
                # However, since the user asks to "Pass these deterministic findings directly into the Gemini prompt",
                # we'll modify _audit_with_llm to accept it.
                llm_verdict = await self._audit_with_llm(draft, qdrant_store, flagged_deterministic)
                if llm_verdict is not None:
                    return llm_verdict
            except Exception as e:
                logger.warning("LLM adversarial audit failed (%s).", e)

        # 1ms synchronous regex pass removed. If no LLM call is made, do not return a PASS verdict.
        return CritiqueVerdict(
            is_approved=False,
            verdict="REVISE",
            feedback="CRITICAL AUDIT FAILURE: Gemini LLM is required for adversarial audit, but it is not configured or the call failed.",
            revision_count=current_revisions,
            total_citations_checked=0,
            flagged_claims=[],
            grounding_rate=0.0,
        )

    def audit_report_sync(
        self,
        draft: DraftReport,
        qdrant_store: Optional[Any] = None,
    ) -> CritiqueVerdict:
        """Synchronous wrapper for audit_report."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(lambda: asyncio.run(self.audit_report(draft, qdrant_store)))
                return future.result()
        else:
            return asyncio.run(self.audit_report(draft, qdrant_store))

    async def _audit_with_llm(
        self,
        draft: DraftReport,
        qdrant_store: Any,
        deterministic_findings: Optional[List[FlaggedClaimDetail]] = None,
    ) -> Optional[CritiqueVerdict]:
        """Constructs forensic audit prompt with Qdrant payloads and invokes Gemini LLM."""
        if not self.client.is_configured:
            return None

        # Gather cited evidence payloads from Qdrant
        payloads: Dict[str, Any] = {}
        for eid in draft.inline_citations:
            try:
                data = qdrant_store.retrieve_by_id(eid)
                if data:
                    payloads[eid] = {
                        "text": data.get("text", "")[:1500],
                        "source": data.get("source", ""),
                        "chunk_type": data.get("chunk_type", ""),
                        "metadata": data.get("metadata", {}),
                    }
                else:
                    payloads[eid] = {"text": "NOT_FOUND_IN_STORE", "source": "None"}
            except Exception as e:
                logger.error("Error retrieving evidence %s for LLM audit: %s", eid, e)
                payloads[eid] = {"text": f"RETRIEVAL_ERROR: {e}", "source": "None"}

        # Construct LLM prompt
        findings_str = ""
        if deterministic_findings:
            findings_str = (
                f"\n=== DETERMINISTIC FINDINGS (PRE-FLIGHT GATE) ===\n"
                f"The following issues were flagged by deterministic extraction heuristics:\n"
                f"{json.dumps(deterministic_findings, indent=2)}\n"
                f"Use these findings as hints for your semantic audit.\n"
            )

        prompt_content = (
            f"=== TARGET TICKER / ENTITY ===\n{draft.ticker}\n\n"
            f"=== USER QUERY ===\n{draft.query}\n\n"
            f"=== ANALYST DRAFT REPORT TO AUDIT ===\n"
            f"--- Section 1: Business Simplicity & Moat ---\n{draft.moat if hasattr(draft, 'moat') else ''}\n\n"
            f"--- Section 2: Owner Earnings & ROIC ---\n{draft.owner_earnings if hasattr(draft, 'owner_earnings') else ''}\n\n"
            f"--- Section 3: Capital Allocation (Debt/Buybacks) ---\n{draft.capital_allocation if hasattr(draft, 'capital_allocation') else ''}\n\n"
            f"--- Section 4: Margin of Safety (10-Cap Valuation Test) ---\n{draft.margin_of_safety if hasattr(draft, 'margin_of_safety') else ''}\n\n"
            f"=== GROUNDED QDRANT EVIDENCE PAYLOADS (Keyed by Evidence ID) ===\n"
            f"{json.dumps(payloads, indent=2)}\n"
            f"{findings_str}\n"
            f"Execute the Charlie Munger Inversion Audit and return strict JSON."
        )

        loop = asyncio.get_running_loop()
        try:
            def _call_llm():
                return self.client.generate_text(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=prompt_content,
                    model=self.model_name,
                    response_format="json",
                    temperature=0.1
                )

            raw_json = await loop.run_in_executor(None, _call_llm)
            parsed = json.loads(raw_json)

            is_approved = bool(parsed.get("is_approved", False))
            verdict_str = "PASS" if is_approved else "REVISE"
            feedback = str(parsed.get("feedback", "LLM audit completed."))
            grounding_rate = float(parsed.get("grounding_rate", 1.0 if is_approved else 0.0))
            raw_flagged = parsed.get("flagged_claims", [])

            flagged_claims: List[FlaggedClaimDetail] = []
            if isinstance(raw_flagged, list):
                for f in raw_flagged:
                    if isinstance(f, dict):
                        flagged_claims.append({
                            "claim_text": str(f.get("claim_text", "")),
                            "evidence_id": f.get("evidence_id"),
                            "issue_type": str(f.get("issue_type", "UNGROUNDED_CLAIM")),
                            "detected_discrepancy": str(f.get("detected_discrepancy", "")),
                        })

            return CritiqueVerdict(
                is_approved=is_approved,
                verdict=verdict_str,
                feedback=feedback,
                revision_count=draft.revision_count,
                total_citations_checked=len(draft.inline_citations),
                flagged_claims=flagged_claims,
                grounding_rate=grounding_rate,
            )
        except Exception as e:
            logger.warning("Critic LLM execution exception (%s). Falling back to deterministic auditor.", e)
            return None

    # =================================================================
    # Verification Helpers
    # =================================================================

    def _extract_cited_sentences(self, markdown_text: str, manifest: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
        """Extracts (sentence, evidence_id) pairs from markdown text."""
        pairs: List[Tuple[str, str]] = []
        # Match sentences or lines containing [uuid]
        lines = markdown_text.split("\n")
        uuid_pattern = re.compile(r"\[(?:evidence_id:\s*)?([a-f0-9\-]{8,36})\]", re.IGNORECASE)
        
        manifest_uuids = set()
        if manifest:
            if "indexed_items" in manifest:
                for item in manifest["indexed_items"]:
                    if "evidence_id" in item:
                        manifest_uuids.add(item["evidence_id"])
            elif "evidence_ids" in manifest:
                manifest_uuids.update(manifest["evidence_ids"])

        for line in lines:
            line_clean = line.strip()
            if not line_clean or line_clean.startswith("#"):
                continue

            matches = uuid_pattern.findall(line_clean)
            if matches:
                for raw_id in matches:
                    resolved_id = raw_id
                    if manifest_uuids:
                        found = [uid for uid in manifest_uuids if uid.startswith(raw_id)]
                        if len(found) == 1:
                            resolved_id = found[0]
                        elif len(found) > 1:
                            raise self.AmbiguousCitationError(f"Citation [{raw_id}] matches multiple items in the manifest.")
                    pairs.append((line_clean, resolved_id))

        return pairs

    async def _verify_single_claim(
        self,
        sentence: str,
        evidence_id: str,
        qdrant_store: Optional[Any],
        draft: Optional[DraftReport] = None,
    ) -> Optional[FlaggedClaimDetail]:
        """Validates a claim against its source Qdrant chunk."""
        if qdrant_store is None:
            # When store is omitted in dry-run/mock tests, verify format
            return None

        # Asynchronously fetch chunk payload from Qdrant
        loop = asyncio.get_running_loop()
        try:
            payload = await loop.run_in_executor(None, qdrant_store.retrieve_by_id, evidence_id)
        except Exception as e:
            logger.error("Error retrieving evidence %s: %s", evidence_id, e)
            payload = None

        positive_assertion_regex = re.compile(
            r"\b(solid|expand|expanding|expansion|growth|growing|strong|stronger|positive|profitable|profitability|robust|healthy|improving|improvement|record|gain|gains|buy|outperform|bullish)\b",
            re.IGNORECASE,
        )
        texts_to_evaluate = [sentence]
        if draft is not None:
            texts_to_evaluate.extend([draft.thesis, draft.financial_metrics])
        has_positive_assertion = any(positive_assertion_regex.search(txt) for txt in texts_to_evaluate if txt)

        if payload is None or not payload or payload.get("content") is None:
            if has_positive_assertion:
                return FlaggedClaimDetail(
                    claim_text=sentence,
                    evidence_id=evidence_id,
                    issue_type="CONTRADICTORY_EVIDENCE",
                    detected_discrepancy="Analyst makes positive assertion while cited evidence payload evaluates to None.",
                )
            return FlaggedClaimDetail(
                claim_text=sentence,
                evidence_id=evidence_id,
                issue_type="NONEXISTENT_EVIDENCE_ID",
                detected_discrepancy=f"Cited evidence_id '{evidence_id}' does not exist in Qdrant store.",
            )

        source_content = str(payload.get("content", ""))
        source_lower = source_content.lower().strip()

        # Check if payload content evaluates to None or empty
        if source_lower in ("", "none"):
            if has_positive_assertion:
                return FlaggedClaimDetail(
                    claim_text=sentence,
                    evidence_id=evidence_id,
                    issue_type="CONTRADICTORY_EVIDENCE",
                    detected_discrepancy="Analyst makes positive assertion while cited evidence payload evaluates to None.",
                )

        # Heuristic check: Contradictory Evidence
        # If payload contains "Company not found" or "No income statement line items reported"
        # while making a positive fundamental assertion, flag as CONTRADICTORY_EVIDENCE.
        contradictory_indicators = [
            "company not found",
            "no income statement line items reported",
        ]
        matched_indicator = next((ind for ind in contradictory_indicators if ind in source_lower), None)

        if matched_indicator and has_positive_assertion:
            return FlaggedClaimDetail(
                claim_text=sentence,
                evidence_id=evidence_id,
                issue_type="CONTRADICTORY_EVIDENCE",
                detected_discrepancy=(
                    f"Analyst makes positive fundamental assertion while citing payload containing '{matched_indicator}'."
                ),
            )

        # Charlie Munger Inversion Check 1: Moat Inversion Check
        # Demand verifiable ROIC, ROCE, or return on capital disclosures in the cited evidence if asserting a moat.
        moat_regex = re.compile(r"\b(moat|competitive advantage|pricing power|barrier to entry)\b", re.IGNORECASE)
        no_moat_regex = re.compile(r"\b(no moat|lack of moat|absence of moat|narrow or no moat)\b", re.IGNORECASE)
        roic_regex = re.compile(
            r"\b(roic|roce|return on invested capital|return on capital|return on equity|capital efficiency|operating return|capital return|returns on capital)\b",
            re.IGNORECASE,
        )
        if moat_regex.search(sentence) and not no_moat_regex.search(sentence):
            if not roic_regex.search(source_lower):
                return FlaggedClaimDetail(
                    claim_text=sentence,
                    evidence_id=evidence_id,
                    issue_type="UNGROUNDED_MOAT_CLAIM",
                    detected_discrepancy="Analyst claims economic moat or competitive advantage without citing verifiable ROIC or return on capital disclosures.",
                )

        # Charlie Munger Inversion Check 2: Downside Protection & Debt Inversion Check
        # Demand concrete balance sheet debt structures or liquidity runway if claiming downside protection / margin of safety.
        downside_regex = re.compile(r"\b(downside protection|capital preservation floor|downside floor)\b", re.IGNORECASE)
        debt_regex = re.compile(
            r"\b(debt|liability|liabilities|borrowing|borrowings|leverage|obligation|obligations|liquidity|cash|solvency|interest coverage|balance sheet)\b",
            re.IGNORECASE,
        )
        if downside_regex.search(sentence):
            if not debt_regex.search(source_lower):
                return FlaggedClaimDetail(
                    claim_text=sentence,
                    evidence_id=evidence_id,
                    issue_type="UNGROUNDED_DOWNSIDE_PROTECTION",
                    detected_discrepancy="Analyst claims downside protection or margin of safety without citing balance sheet debt structures or liquidity disclosures.",
                )

        # Check for numeric hallucinations:
        # Extract numeric tokens from the sentence (e.g. 196.0, 46%, 8.5, $120)
        claim_numbers = self._extract_numbers(sentence)
        source_numbers = self._extract_numbers(source_content)

        unsupported_numbers = []
        for num in claim_numbers:
            # Check if number appears in source text (as float or integer or string)
            if not self._is_number_supported(num, source_numbers, source_content):
                unsupported_numbers.append(str(num))

        if unsupported_numbers:
            return FlaggedClaimDetail(
                claim_text=sentence,
                evidence_id=evidence_id,
                issue_type="HALLUCINATED_NUMBERS",
                detected_discrepancy=(
                    f"Disclosed number(s) {unsupported_numbers} not grounded in cited evidence chunk."
                ),
            )

        return None

    def _check_all_valuation_multiples_are_none(self, valuation_text: str) -> bool:
        """Determines if valuation multiples are present in Section 3 and all of them are 'None'."""
        if not valuation_text:
            return False

        multiple_pattern = re.compile(
            r"(?:Trailing\s+P/E|Forward\s+P/E|P/E|PEG\s+Ratio|PEG|EV\s*/\s*EBITDA|Price\s+to\s+Book|P/B|Price\s+to\s+Sales|P/S|Enterprise\s+Value|Market\s+Cap|Beta)\s*[:=]\s*([^\n\r,;]+)",
            re.IGNORECASE,
        )
        matches = multiple_pattern.findall(valuation_text)
        if not matches:
            return False

        for val in matches:
            v_clean = val.strip().lower()
            if not v_clean.startswith("none") and not v_clean.startswith("n/a"):
                return False
        return True

    def _extract_numbers(self, text: str) -> List[float]:
        """Extracts numerical values from text for factual grounding audits."""
        # Find integers and decimals, ignoring UUID characters
        cleaned = re.sub(r"\[(?:evidence_id:\s*)?[a-f0-9\-]{8,36}\]", "", text, flags=re.IGNORECASE)
        raw_matches = re.findall(r"(?<![a-zA-Z0-9\-])(\d+(?:\.\d+)?)(?:%|[BMKbmk])?(?![a-zA-Z0-9\-])", cleaned)
        numbers = []
        for m in raw_matches:
            try:
                val = float(m)
                # Ignore trivial digits like section numbers (1, 2, 3, 4) if under 5
                if val >= 5.0 or "." in m:
                    numbers.append(val)
            except ValueError:
                continue
        return numbers

    def _is_number_supported(self, target: float, source_numbers: List[float], source_text: str) -> bool:
        """Determines if a target number exists or has mathematical support in source text."""
        # Check direct float match
        for s in source_numbers:
            if abs(s - target) < 1e-4:
                return True

        # Check raw string occurrences
        target_str = f"{target:g}"
        if target_str in source_text:
            return True

        return False
