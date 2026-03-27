"""
Layer 2 – Multi-Timeframe Trend Analysis.

Calculates Supertrend(10,3) and EMA stack (20/50/200) for each timeframe.
All indicators implemented from scratch using numpy only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Indicator helpers (numpy-only)
# ──────────────────────────────────────────────────────────


def _to_arrays(candles: List[Dict]) -> tuple:
    """Extract high, low, close arrays from candle list."""
    highs = np.array([c["high"] for c in candles], dtype=np.float64)
    lows = np.array([c["low"] for c in candles], dtype=np.float64)
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    return highs, lows, closes


def _atr_wilder(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    """ATR using Wilder smoothing."""
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


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result
    result[period - 1] = np.mean(values[:period])
    k = 2.0 / (period + 1)
    for i in range(period, n):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def _supertrend(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> np.ndarray:
    """
    Calculate Supertrend.
    Returns array of 1.0 (UP/bullish) or -1.0 (DOWN/bearish), NaN where not yet computed.
    """
    n = len(closes)
    atr = _atr_wilder(highs, lows, closes, period)

    hl2 = (highs + lows) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    final_upper = np.full(n, np.nan, dtype=np.float64)
    final_lower = np.full(n, np.nan, dtype=np.float64)
    supertrend = np.full(n, np.nan, dtype=np.float64)

    for i in range(period, n):
        # Final upper band
        if np.isnan(final_upper[i - 1]) or upper_band[i] < final_upper[i - 1]:
            final_upper[i] = upper_band[i]
        elif closes[i - 1] > final_upper[i - 1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band
        if np.isnan(final_lower[i - 1]) or lower_band[i] > final_lower[i - 1]:
            final_lower[i] = lower_band[i]
        elif closes[i - 1] < final_lower[i - 1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Supertrend direction
        prev_st = supertrend[i - 1]
        if np.isnan(prev_st):
            prev_st = -1.0  # default UP initially

        if prev_st == final_upper[i - 1] or np.isnan(supertrend[i - 1]):
            # Was bearish
            if closes[i] > final_upper[i]:
                supertrend[i] = final_lower[i]  # flip to bullish
            else:
                supertrend[i] = final_upper[i]  # stay bearish
        else:
            # Was bullish
            if closes[i] < final_lower[i]:
                supertrend[i] = final_upper[i]  # flip to bearish
            else:
                supertrend[i] = final_lower[i]  # stay bullish

    # Direction: 1 if close > supertrend (UP), -1 if close < supertrend (DOWN)
    direction = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if not np.isnan(supertrend[i]):
            direction[i] = 1.0 if closes[i] > supertrend[i] else -1.0

    return direction


def _supertrend_signal(candles: List[Dict], period: int = 10, mult: float = 3.0) -> Optional[str]:
    """Return 'UP' or 'DOWN' for the latest bar, or None if insufficient data."""
    if not candles or len(candles) < period + 5:
        return None
    try:
        highs, lows, closes = _to_arrays(candles)
        direction = _supertrend(highs, lows, closes, period, mult)
        valid = direction[~np.isnan(direction)]
        if len(valid) == 0:
            return None
        return "UP" if valid[-1] > 0 else "DOWN"
    except Exception as exc:
        logger.warning("Supertrend error: %s", exc)
        return None


def _ema_stack_bullish(candles: List[Dict]) -> Optional[bool]:
    """
    True if EMA20 > EMA50 > EMA200 (bullish stack).
    False if EMA20 < EMA50 < EMA200 (bearish stack).
    None if mixed/insufficient.
    """
    if not candles or len(candles) < 205:
        return None
    try:
        closes = np.array([c["close"] for c in candles], dtype=np.float64)
        ema20 = _ema(closes, 20)
        ema50 = _ema(closes, 50)
        ema200 = _ema(closes, 200)

        e20 = ema20[-1]
        e50 = ema50[-1]
        e200 = ema200[-1]

        if any(np.isnan(x) for x in [e20, e50, e200]):
            return None

        if e20 > e50 > e200:
            return True
        if e20 < e50 < e200:
            return False
        return None  # mixed
    except Exception as exc:
        logger.warning("EMA stack error: %s", exc)
        return None


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────


@dataclass
class TrendResult:
    aligned_count: int          # 0-3 timeframes aligned
    direction: str              # "LONG", "SHORT", "MIXED"
    supertrend_signals: Dict[str, str]   # {"4h": "UP"/"DOWN", ...}
    ema_aligned: Dict[str, bool]         # {"4h": True/False, ...}
    score: float                # 0-25
    details: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────
# Main analysis function
# ──────────────────────────────────────────────────────────


def analyze(candles_by_tf: Dict[str, List[Dict]]) -> TrendResult:
    """
    Multi-timeframe trend analysis.

    candles_by_tf: dict mapping timeframe string -> list of OHLCV dicts.
    Expected keys: "4h", "1h", "15m" (any subset is acceptable).
    """
    _neutral = TrendResult(
        aligned_count=0,
        direction="MIXED",
        supertrend_signals={},
        ema_aligned={},
        score=0.0,
        details={"error": "insufficient data"},
    )

    if not candles_by_tf:
        return _neutral

    supertrend_signals: Dict[str, Optional[str]] = {}
    ema_signals: Dict[str, Optional[bool]] = {}

    for tf, candles in candles_by_tf.items():
        if not candles:
            supertrend_signals[tf] = None
            ema_signals[tf] = None
            continue
        supertrend_signals[tf] = _supertrend_signal(candles)
        ema_signals[tf] = _ema_stack_bullish(candles)

    # Determine how many TFs have the same directional signal
    st_up = [tf for tf, v in supertrend_signals.items() if v == "UP"]
    st_down = [tf for tf, v in supertrend_signals.items() if v == "DOWN"]

    # Overall direction from supertrend majority
    if len(st_up) > len(st_down):
        primary_direction = "LONG"
    elif len(st_down) > len(st_up):
        primary_direction = "SHORT"
    else:
        primary_direction = "MIXED"

    # Count aligned TFs (Supertrend and EMA agree)
    aligned_tfs: List[str] = []
    ema_aligned_map: Dict[str, bool] = {}

    for tf in candles_by_tf.keys():
        st = supertrend_signals.get(tf)
        ema = ema_signals.get(tf)

        if st is None:
            ema_aligned_map[tf] = False
            continue

        ema_bullish = ema  # True/False/None
        st_bullish = st == "UP"

        # EMA stack aligned with ST
        if ema_bullish is None:
            # Can't confirm EMA; partial alignment only
            ema_aligned_map[tf] = False
        elif st_bullish == ema_bullish:
            ema_aligned_map[tf] = True
            aligned_tfs.append(tf)
        else:
            ema_aligned_map[tf] = False

    aligned_count = len(aligned_tfs)

    # Check direct conflict: UP and DOWN in same candles
    has_conflict = len(st_up) > 0 and len(st_down) > 0

    # Score computation
    if aligned_count == 3:
        score = 25.0
    elif aligned_count == 2:
        # Check if they're pointing same direction
        aligned_directions = [supertrend_signals.get(tf) for tf in aligned_tfs]
        if len(set(aligned_directions)) == 1:
            score = 18.0
        else:
            score = 10.0
    elif aligned_count == 1:
        score = 5.0
    else:
        score = 0.0

    # Penalty for direct Supertrend conflicts across TFs
    if has_conflict and len(st_up) > 0 and len(st_down) > 0:
        score = max(0.0, score - 5.0)

    # Build clean ema_aligned output (only True entries)
    ema_aligned_clean = {tf: bool(v) for tf, v in ema_aligned_map.items() if v is not None}

    # Filter out None from supertrend_signals
    st_clean = {tf: v for tf, v in supertrend_signals.items() if v is not None}

    details = {
        "supertrend_by_tf": st_clean,
        "ema_stack_by_tf": {tf: v for tf, v in ema_signals.items()},
        "aligned_tfs": aligned_tfs,
        "has_conflict": has_conflict,
        "st_up_tfs": st_up,
        "st_down_tfs": st_down,
        "score_raw": score,
    }

    return TrendResult(
        aligned_count=aligned_count,
        direction=primary_direction,
        supertrend_signals=st_clean,
        ema_aligned=ema_aligned_clean,
        score=round(score, 2),
        details=details,
    )
