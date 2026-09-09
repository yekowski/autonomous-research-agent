"""Financial Market Data Tool.

Wrapper around yfinance providing structured, strictly-typed access to historical prices,
financial statements, key valuation multiples, and recent news feeds with robust error handling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore


# =====================================================================
# Strict TypedDict Definitions
# =====================================================================

class ValuationMultiples(TypedDict):
    """Strictly typed key valuation multiples and pricing parameters."""
    symbol: str
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    peg_ratio: Optional[float]
    enterprise_to_ebitda: Optional[float]
    price_to_book: Optional[float]
    price_to_sales: Optional[float]
    enterprise_value: Optional[float]
    market_cap: Optional[float]
    beta: Optional[float]
    currency: str


class FinancialStatementsData(TypedDict):
    """Strictly typed financial statement output."""
    symbol: str
    statement_type: str  # 'income' | 'balance_sheet' | 'cashflow'
    period_type: str     # 'annual' | 'quarterly'
    currency: str
    metrics: Dict[str, Dict[str, Optional[float]]]  # metric_name -> {period_date: value}


class NewsItem(TypedDict):
    """Strictly typed corporate news item."""
    id: str
    symbol: str
    title: str
    publisher: str
    link: str
    published_at: Optional[str]
    summary: Optional[str]


class CompanyProfile(TypedDict):
    """Strictly typed company profile and valuation summary."""
    symbol: str
    short_name: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    market_cap: Optional[float]
    forward_pe: Optional[float]
    trailing_pe: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    currency: str


# =====================================================================
# Custom Exceptions
# =====================================================================

class FinancialDataError(Exception):
    """Base exception for financial data operations."""
    pass


class TickerNotFoundError(FinancialDataError):
    """Raised when the specified ticker cannot be found or is delisted."""
    pass


class InsufficientDataError(FinancialDataError):
    """Raised when statement tables are empty and all valuation multiples are None."""
    pass


class FinancialDataFetchError(FinancialDataError):
    """Raised when an API or network error occurs while fetching financial data."""
    pass


# =====================================================================
# Tool Implementation
# =====================================================================

class FinancialDataTool:
    """Production-grade interface for quantitative market and corporate financial data."""

    def __init__(self) -> None:
        pass

    def _ensure_yfinance(self) -> None:
        """Validates that yfinance is installed in the current environment."""
        if yf is None:
            raise ImportError(
                "yfinance is required for FinancialDataTool. "
                "Install project dependencies with: pip install yfinance"
            )

    def validate_ticker(self, ticker: str, t: Optional[Any] = None) -> None:
        """Pre-checks that the ticker has active market data using fast_info.
        
        Raises:
            TickerNotFoundError: If market_cap or quote_type is unavailable/empty.
        """
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            raise ValueError("Ticker symbol cannot be empty.")
        self._ensure_yfinance()
        if t is None:
            t = yf.Ticker(clean_ticker)

        try:
            fast_info = getattr(t, "fast_info", None)
            if fast_info is None:
                raise TickerNotFoundError(f"Ticker '{clean_ticker}' has no active market data.")

            market_cap = getattr(fast_info, "market_cap", None)
            if market_cap is None and hasattr(fast_info, "get"):
                market_cap = fast_info.get("market_cap")

            quote_type = getattr(fast_info, "quote_type", None)
            if quote_type is None and hasattr(fast_info, "get"):
                quote_type = fast_info.get("quote_type")

            if market_cap is None or market_cap == "" or not quote_type:
                raise TickerNotFoundError(f"Ticker '{clean_ticker}' has no active market data.")
        except TickerNotFoundError:
            raise
        except Exception as e:
            raise TickerNotFoundError(f"Ticker '{clean_ticker}' has no active market data.") from e

    def check_financial_data_sufficiency(self, ticker: str, t: Optional[Any] = None) -> None:
        """Pre-checks that the ticker returns statement tables or valuation multiples.
        
        Raises:
            InsufficientDataError: If statement tables are empty and all valuation multiples are None.
        """
        clean_ticker = ticker.strip().upper()
        self._ensure_yfinance()
        if t is None:
            t = yf.Ticker(clean_ticker)

        has_statements = False
        try:
            inc = getattr(t, "income_stmt", None)
            bal = getattr(t, "balance_sheet", None)
            if inc is not None and not inc.empty:
                has_statements = True
            elif bal is not None and not bal.empty:
                has_statements = True
        except Exception:
            pass

        has_multiples = False
        try:
            info = getattr(t, "info", None) or {}
            numeric_fields = [
                "trailingPE", "forwardPE", "pegRatio", "enterpriseToEbitda",
                "priceToBook", "priceToSalesTrailing12Months", "enterpriseValue",
                "marketCap", "beta"
            ]
            if any(info.get(k) is not None for k in numeric_fields):
                has_multiples = True
        except Exception:
            pass

        if not has_statements and not has_multiples:
            raise InsufficientDataError("No financial data found.")

    def _get_ticker_obj(self, ticker: str) -> Any:
        """Instantiates yfinance.Ticker with validation."""
        clean_ticker = ticker.strip().upper()
        if not clean_ticker:
            raise ValueError("Ticker symbol cannot be empty.")
        self._ensure_yfinance()
        t = yf.Ticker(clean_ticker)
        self.validate_ticker(clean_ticker, t)
        self.check_financial_data_sufficiency(clean_ticker, t)
        return clean_ticker, t

    def get_valuation_multiples(self, ticker: str) -> ValuationMultiples:
        """Retrieves key valuation multiples (P/E, forward P/E, EV/EBITDA, etc.) as a strictly typed dict.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'NVDA', 'TRANSCORP.LG').

        Returns:
            ValuationMultiples: Typed dictionary containing normalized valuation ratios.

        Raises:
            TickerNotFoundError: If the ticker is invalid or returns no data.
            FinancialDataFetchError: If the remote fetch fails.
        """
        symbol, t = self._get_ticker_obj(ticker)
        try:
            info = t.info or {}
        except Exception as e:
            logger.error("Failed to fetch info for %s: %s", symbol, e)
            raise FinancialDataFetchError(f"Failed to fetch market data for {symbol}: {e}") from e

        if not info or info.get("trailingPegRatio") is None and info.get("regularMarketPrice") is None and not info.get("shortName"):
            # If info is effectively empty or contains error indicators
            if len(info) <= 1:
                raise TickerNotFoundError(f"Ticker '{symbol}' not found or has no available market data.")

        def _safe_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                f = float(val)
                import math
                return None if math.isnan(f) else f
            except (ValueError, TypeError):
                return None

        return ValuationMultiples(
            symbol=symbol,
            trailing_pe=_safe_float(info.get("trailingPE")),
            forward_pe=_safe_float(info.get("forwardPE")),
            peg_ratio=_safe_float(info.get("pegRatio")),
            enterprise_to_ebitda=_safe_float(info.get("enterpriseToEbitda")),
            price_to_book=_safe_float(info.get("priceToBook")),
            price_to_sales=_safe_float(info.get("priceToSalesTrailing12Months")),
            enterprise_value=_safe_float(info.get("enterpriseValue")),
            market_cap=_safe_float(info.get("marketCap")),
            beta=_safe_float(info.get("beta")),
            currency=str(info.get("currency", "USD")),
        )

    def get_income_statement(self, ticker: str, quarterly: bool = False) -> FinancialStatementsData:
        """Retrieves income statement data as a strictly typed dictionary.
        
        Args:
            ticker: Stock ticker symbol.
            quarterly: Whether to retrieve quarterly statements (default: annual).
        """
        return self._fetch_statement(ticker, statement_type="income", quarterly=quarterly)

    def get_balance_sheet(self, ticker: str, quarterly: bool = False) -> FinancialStatementsData:
        """Retrieves balance sheet data as a strictly typed dictionary.
        
        Args:
            ticker: Stock ticker symbol.
            quarterly: Whether to retrieve quarterly statements (default: annual).
        """
        return self._fetch_statement(ticker, statement_type="balance_sheet", quarterly=quarterly)

    def _fetch_statement(
        self,
        ticker: str,
        statement_type: str,
        quarterly: bool = False,
    ) -> FinancialStatementsData:
        """Fetches and normalizes financial statement DataFrames into typed dictionaries."""
        symbol, t = self._get_ticker_obj(ticker)
        try:
            if statement_type == "income":
                df = t.quarterly_income_stmt if quarterly else t.income_stmt
            elif statement_type == "balance_sheet":
                df = t.quarterly_balance_sheet if quarterly else t.balance_sheet
            elif statement_type == "cashflow":
                df = t.quarterly_cashflow if quarterly else t.cashflow
            else:
                raise ValueError(f"Unknown statement type: {statement_type}")
        except Exception as e:
            logger.error("Error fetching %s statement for %s: %s", statement_type, symbol, e)
            raise FinancialDataFetchError(f"Failed to fetch {statement_type} statement for {symbol}: {e}") from e

        metrics_data: Dict[str, Dict[str, Optional[float]]] = {}
        if df is not None and not df.empty:
            for metric_idx, row in df.iterrows():
                metric_name = str(metric_idx)
                metrics_data[metric_name] = {}
                for col_date, val in row.items():
                    date_str = str(col_date)[:10]  # YYYY-MM-DD
                    try:
                        import math
                        fval = float(val)
                        metrics_data[metric_name][date_str] = None if math.isnan(fval) else fval
                    except (ValueError, TypeError):
                        metrics_data[metric_name][date_str] = None

        currency = "USD"
        try:
            currency = str(t.info.get("currency", "USD")) if t.info else "USD"
        except Exception:
            pass

        return FinancialStatementsData(
            symbol=symbol,
            statement_type=statement_type,
            period_type="quarterly" if quarterly else "annual",
            currency=currency,
            metrics=metrics_data,
        )

    def get_recent_news(self, ticker: str, limit: int = 10) -> List[NewsItem]:
        """Retrieves recent news articles for the company as strictly typed NewsItem dictionaries."""
        symbol, t = self._get_ticker_obj(ticker)
        try:
            raw_news = t.news or []
        except Exception as e:
            logger.error("Error fetching news for %s: %s", symbol, e)
            raise FinancialDataFetchError(f"Failed to fetch news for {symbol}: {e}") from e

        results: List[NewsItem] = []
        for idx, item in enumerate(raw_news[:limit]):
            # yfinance news payloads can vary slightly across versions
            content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
            title = item.get("title") or content.get("title") or "Corporate News"
            pub = item.get("publisher") or content.get("provider", {}).get("displayName") or "Financial Media"
            link = item.get("link") or content.get("canonicalUrl", {}).get("url") or ""
            pub_time = str(item.get("providerPublishTime") or content.get("pubDate") or "")
            summary = item.get("summary") or content.get("summary")

            results.append(
                NewsItem(
                    id=str(item.get("uuid") or f"news_{symbol}_{idx}"),
                    symbol=symbol,
                    title=str(title),
                    publisher=str(pub),
                    link=str(link),
                    published_at=pub_time,
                    summary=str(summary) if summary else None,
                )
            )

        return results

    def get_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Public alias that satisfies the SKILLS.md contract exactly by returning raw dicts."""
        return [dict(n) for n in self.get_recent_news(ticker, limit)]

    def get_ticker_summary(self, ticker: str) -> CompanyProfile:
        """Retrieves core profile and valuation summary for a ticker symbol."""
        symbol, t = self._get_ticker_obj(ticker)
        try:
            info = t.info or {}
        except Exception as e:
            raise FinancialDataFetchError(f"Failed to fetch ticker info for {symbol}: {e}") from e

        def _safe_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                import math
                f = float(val)
                return None if math.isnan(f) else f
            except (ValueError, TypeError):
                return None

        return CompanyProfile(
            symbol=symbol,
            short_name=info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            market_cap=_safe_float(info.get("marketCap")),
            forward_pe=_safe_float(info.get("forwardPE")),
            trailing_pe=_safe_float(info.get("trailingPE")),
            fifty_two_week_high=_safe_float(info.get("fiftyTwoWeekHigh")),
            fifty_two_week_low=_safe_float(info.get("fiftyTwoWeekLow")),
            currency=str(info.get("currency", "USD")),
        )
