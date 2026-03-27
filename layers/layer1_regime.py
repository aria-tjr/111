"""
Layer 1 – Market Regime Detection.

Calculates ADX(14) and ATR(14) from scratch using numpy.
No pandas or ta-lib dependency.
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


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using Wilder's smoothing.
    Returns array same length as input (first `period` values are NaN).
    """
    n = len(closes)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

    atr_arr = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return atr_arr

    # Seed with simple average
    atr_arr[period - 1] = np.mean(tr[:period])
    # Wilder smoothing
    k = 1.0 / period
    for i in range(period, n):
        atr_arr[i] = atr_arr[i - 1] * (1 - k) + tr[i] * k

    return atr_arr


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> tuple:
    """
    Calculate +DI, -DI and ADX.
    Returns (adx_arr, plus_di, minus_di) all same length as input.
    """
    n = len(closes)
    tr = np.empty(n, dtype=np.float64)
    pdm = np.empty(n, dtype=np.float64)
    ndm = np.empty(n, dtype=np.float64)

    tr[0] = highs[0] - lows[0]
    pdm[0] = 0.0
    ndm[0] = 0.0

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)

        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        if up > down and up > 0:
            pdm[i] = up
        else:
            pdm[i] = 0.0
        if down > up and down > 0:
            ndm[i] = down
        else:
            ndm[i] = 0.0

    adx_arr = np.full(n, np.nan, dtype=np.float64)
    plus_di_arr = np.full(n, np.nan, dtype=np.float64)
    minus_di_arr = np.full(n, np.nan, dtype=np.float64)

    if n < 2 * period:
        return adx_arr, plus_di_arr, minus_di_arr

    # Wilder smoothed sums
    smtr = np.full(n, np.nan, dtype=np.float64)
    smpdm = np.full(n, np.nan, dtype=np.float64)
    smndm = np.full(n, np.nan, dtype=np.float64)

    smtr[period - 1] = np.sum(tr[:period])
    smpdm[period - 1] = np.sum(pdm[:period])
    smndm[period - 1] = np.sum(ndm[:period])

    for i in range(period, n):
        smtr[i] = smtr[i - 1] - smtr[i - 1] / period + tr[i]
        smpdm[i] = smpdm[i - 1] - smpdm[i - 1] / period + pdm[i]
        smndm[i] = smndm[i - 1] - smndm[i - 1] / period + ndm[i]

    for i in range(period - 1, n):
        if smtr[i] == 0:
            continue
        pdi = 100.0 * smpdm[i] / smtr[i]
        ndi = 100.0 * smndm[i] / smtr[i]
        plus_di_arr[i] = pdi
        minus_di_arr[i] = ndi

    # DX
    dx = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        pdi = plus_di_arr[i]
        ndi = minus_di_arr[i]
        if np.isnan(pdi) or np.isnan(ndi):
            continue
        denom = pdi + ndi
        if denom == 0:
            dx[i] = 0.0
        else:
            dx[i] = 100.0 * abs(pdi - ndi) / denom

    # Smooth DX -> ADX (Wilder, period bars)
    first_valid = period - 1 + period - 1  # index of first valid DX we can seed from
    if first_valid >= n:
        return adx_arr, plus_di_arr, minus_di_arr

    seed_start = period - 1
    seed_end = seed_start + period
    if seed_end > n:
        return adx_arr, plus_di_arr, minus_di_arr

    valid_dx = dx[seed_start:seed_end]
    if np.any(np.isnan(valid_dx)):
        return adx_arr, plus_di_arr, minus_di_arr

    adx_arr[seed_end - 1] = np.nanmean(valid_dx)
    for i in range(seed_end, n):
        if np.isnan(dx[i]):
            adx_arr[i] = adx_arr[i - 1]
        else:
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

    return adx_arr, plus_di_arr, minus_di_arr


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────


@dataclass
class RegimeResult:
    is_trending: bool
    adx: float
    atr_percentile: float  # 0-100
    volatility_state: str  # "low", "normal", "high"
    score: float           # 0-15
    details: Dict[str, Any] = field(default_factory=dict)
    atr_value: float = 0.0  # raw ATR of last bar


# ──────────────────────────────────────────────────────────
# Main analysis function
# ──────────────────────────────────────────────────────────


def analyze(candles: List[Dict]) -> RegimeResult:
    """
    Analyse market regime from OHLCV candle list (15m preferred).

    Returns a RegimeResult with score 0-15.
    """
    # Defensive: handle empty/short input
    _neutral = RegimeResult(
        is_trending=False,
        adx=0.0,
        atr_percentile=50.0,
        volatility_state="normal",
        score=0.0,
        details={"error": "insufficient data"},
        atr_value=0.0,
    )

    if not candles or len(candles) < 30:
        logger.warning("Layer1: insufficient candles (%d), returning neutral", len(candles) if candles else 0)
        return _neutral

    try:
        highs, lows, closes = _to_arrays(candles)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Layer1: candle parsing error: %s", exc)
        return _neutral

    period = 14

    # ── ATR ──────────────────────────────────────────────
    atr_arr = _atr(highs, lows, closes, period=period)
    last_atr = float(np.nanmean(atr_arr[-3:])) if not np.all(np.isnan(atr_arr)) else 0.0

    # ATR percentile over the last 100 bars
    lookback = min(100, len(atr_arr))
    recent_atr = atr_arr[-lookback:]
    valid_atr = recent_atr[~np.isnan(recent_atr)]

    if len(valid_atr) < 5:
        atr_pct = 50.0
    else:
        current_atr = valid_atr[-1]
        atr_pct = float(np.sum(valid_atr <= current_atr) / len(valid_atr) * 100)

    # Volatility state
    if atr_pct < 25:
        vol_state = "low"
    elif atr_pct > 75:
        vol_state = "high"
    else:
        vol_state = "normal"

    # ── ADX ──────────────────────────────────────────────
    adx_arr, plus_di, minus_di = _adx(highs, lows, closes, period=period)

    # Use average of last 3 valid bars for stability
    valid_adx = adx_arr[~np.isnan(adx_arr)]
    if len(valid_adx) == 0:
        logger.warning("Layer1: could not compute ADX")
        return _neutral

    adx_val = float(np.nanmean(adx_arr[-3:]) if len(valid_adx) >= 3 else valid_adx[-1])

    # ── Trend gate ────────────────────────────────────────
    is_trending = adx_val >= 20

    # ── Score ─────────────────────────────────────────────
    if adx_val < 15:
        score = 0.0
    elif adx_val < 20:
        score = 3.0
    elif adx_val >= 25 and vol_state == "normal":
        score = 15.0
    elif adx_val >= 25 and vol_state == "low":
        score = 12.0
    elif adx_val >= 25 and vol_state == "high":
        score = 10.0
    elif adx_val >= 20 and vol_state == "normal":
        score = 12.0
    elif adx_val >= 20:
        score = 8.0
    else:
        score = 5.0

    details = {
        "adx": round(adx_val, 2),
        "atr": round(last_atr, 6),
        "atr_percentile": round(atr_pct, 1),
        "volatility_state": vol_state,
        "is_trending": is_trending,
        "plus_di": round(float(np.nanmean(plus_di[-3:])) if not np.all(np.isnan(plus_di)) else 0, 2),
        "minus_di": round(float(np.nanmean(minus_di[-3:])) if not np.all(np.isnan(minus_di)) else 0, 2),
    }

    return RegimeResult(
        is_trending=is_trending,
        adx=round(adx_val, 2),
        atr_percentile=round(atr_pct, 1),
        volatility_state=vol_state,
        score=score,
        details=details,
        atr_value=round(last_atr, 8),
    )
