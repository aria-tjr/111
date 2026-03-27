"""
Layer 6 – Session and Correlation Filter.

Detects trading session, quality, and BTC correlation alignment.
Supertrend computed from scratch using numpy.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Session definitions (UTC hours)
# ──────────────────────────────────────────────────────────

# (name, start_hour_inclusive, end_hour_exclusive, quality)
SESSIONS = [
    ("DEAD",     21, 24, "LOW"),    # 21:00-24:00 UTC
    ("DEAD",      0,  1, "LOW"),    # 00:00-01:00 UTC
    ("ASIA",      1,  7, "MEDIUM"),
    ("LONDON",    7, 12, "MEDIUM"),
    ("OVERLAP",  12, 16, "HIGH"),   # London + NY overlap
    ("NEW_YORK", 16, 21, "MEDIUM"),
]


def _get_session(utc_hour: int) -> tuple:
    """Return (session_name, quality) for given UTC hour."""
    for name, start, end, quality in SESSIONS:
        if start <= utc_hour < end:
            return name, quality
    return "DEAD", "LOW"


# ──────────────────────────────────────────────────────────
# Supertrend (reused from layer2, minimal copy)
# ──────────────────────────────────────────────────────────


def _atr_wilder(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> np.ndarray:
    n = len(closes)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


def _supertrend_direction(candles: List[Dict], period: int = 10, mult: float = 3.0) -> Optional[str]:
    """Return 'UP' or 'DOWN' for the latest candle, or None."""
    if not candles or len(candles) < period + 5:
        return None
    try:
        highs = np.array([c["high"] for c in candles], dtype=np.float64)
        lows = np.array([c["low"] for c in candles], dtype=np.float64)
        closes = np.array([c["close"] for c in candles], dtype=np.float64)

        n = len(closes)
        atr = _atr_wilder(highs, lows, closes, period)
        hl2 = (highs + lows) / 2.0
        upper_band = hl2 + mult * atr
        lower_band = hl2 - mult * atr

        final_upper = np.full(n, np.nan, dtype=np.float64)
        final_lower = np.full(n, np.nan, dtype=np.float64)
        direction = np.full(n, np.nan, dtype=np.float64)

        for i in range(period, n):
            if np.isnan(final_upper[i - 1]) or upper_band[i] < final_upper[i - 1]:
                final_upper[i] = upper_band[i]
            elif closes[i - 1] > final_upper[i - 1]:
                final_upper[i] = upper_band[i]
            else:
                final_upper[i] = final_upper[i - 1]

            if np.isnan(final_lower[i - 1]) or lower_band[i] > final_lower[i - 1]:
                final_lower[i] = lower_band[i]
            elif closes[i - 1] < final_lower[i - 1]:
                final_lower[i] = lower_band[i]
            else:
                final_lower[i] = final_lower[i - 1]

            prev_dir = direction[i - 1]
            if np.isnan(prev_dir):
                prev_dir = -1.0

            if prev_dir == final_upper[i - 1] or np.isnan(direction[i - 1]):
                st_val = final_lower[i] if closes[i] > final_upper[i] else final_upper[i]
            else:
                st_val = final_upper[i] if closes[i] < final_lower[i] else final_lower[i]

            direction[i] = st_val

        valid = direction[~np.isnan(direction)]
        if len(valid) == 0:
            return None

        last_st = valid[-1]
        last_close = closes[-1]
        return "UP" if last_close > last_st else "DOWN"

    except Exception as exc:
        logger.warning("Layer6 supertrend error: %s", exc)
        return None


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────


@dataclass
class SessionResult:
    current_session: str     # "ASIA", "LONDON", "NEW_YORK", "OVERLAP", "DEAD"
    session_quality: str     # "HIGH", "MEDIUM", "LOW"
    btc_correlation_ok: bool
    score: float             # 0-5
    details: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────
# Main analysis function
# ──────────────────────────────────────────────────────────


def analyze(
    symbol: str,
    candles_15m: List[Dict],
    btc_candles: Optional[List[Dict]] = None,
    direction_hint: str = "LONG",
    utc_now: Optional[datetime.datetime] = None,
) -> SessionResult:
    """
    Session and correlation filter.

    symbol: trading pair (e.g. "ETHUSDT")
    candles_15m: 15m OHLCV for the symbol (used to confirm direction)
    btc_candles: 15m BTC OHLCV (only checked if symbol != BTCUSDT)
    direction_hint: "LONG" or "SHORT"
    utc_now: override current UTC time (for testing)
    """
    _neutral = SessionResult(
        current_session="DEAD",
        session_quality="LOW",
        btc_correlation_ok=False,
        score=0.0,
        details={"error": "insufficient data"},
    )

    # Current UTC time
    if utc_now is None:
        utc_now = datetime.datetime.now(datetime.timezone.utc)

    utc_hour = utc_now.hour
    session_name, quality = _get_session(utc_hour)

    # ── Base score ────────────────────────────────────────
    quality_score_map = {"HIGH": 5, "MEDIUM": 3, "LOW": 1}
    if session_name == "DEAD":
        base_score = 0.0
    else:
        base_score = float(quality_score_map.get(quality, 1))

    # ── BTC Correlation ──────────────────────────────────
    is_btc = symbol.upper() in ("BTCUSDT", "BTCPERP", "XBTUSDT")

    if is_btc:
        btc_correlation_ok = True
        btc_st = None
    else:
        btc_st = _supertrend_direction(btc_candles) if btc_candles else None

        if btc_st is None:
            # No BTC data — treat as neutral (don't penalize)
            btc_correlation_ok = True
        else:
            signal_up = direction_hint.upper() == "LONG"
            btc_up = btc_st == "UP"
            btc_correlation_ok = signal_up == btc_up

    # Bonus for BTC alignment, capped at max 5
    if btc_correlation_ok and not is_btc and btc_candles:
        bonus = 2.0
    else:
        bonus = 0.0

    score = min(5.0, base_score + bonus)

    # If dead session, force score to 0
    if session_name == "DEAD":
        score = 0.0

    details = {
        "utc_hour": utc_hour,
        "session": session_name,
        "quality": quality,
        "base_score": base_score,
        "btc_supertrend": btc_st,
        "btc_correlation_ok": btc_correlation_ok,
        "btc_bonus": bonus,
        "symbol": symbol,
        "direction_hint": direction_hint,
    }

    return SessionResult(
        current_session=session_name,
        session_quality=quality,
        btc_correlation_ok=btc_correlation_ok,
        score=round(score, 2),
        details=details,
    )
