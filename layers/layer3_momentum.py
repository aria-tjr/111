"""
Layer 3 – Momentum Quality Analysis.

Calculates StochRSI, MACD, and RSI divergence from scratch using numpy.
No pandas or ta-lib dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Indicator helpers (numpy-only)
# ──────────────────────────────────────────────────────────


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI."""
    n = len(closes)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return result

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 1e9
        result[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    # Seed the first value
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = np.mean(gains[:period]) / np.mean(losses[:period])
        result[period] = 100.0 - 100.0 / (1.0 + rs)

    return result


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA using numpy."""
    n = len(values)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return result
    result[period - 1] = np.mean(values[:period])
    k = 2.0 / (period + 1)
    for i in range(period, n):
        if np.isnan(values[i]):
            result[i] = result[i - 1]
        else:
            result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def _stochrsi(
    closes: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic RSI.
    Returns (K, D) arrays.
    """
    rsi_vals = _rsi(closes, rsi_period)
    n = len(closes)
    stoch_k_raw = np.full(n, np.nan, dtype=np.float64)

    for i in range(stoch_period - 1, n):
        window = rsi_vals[i - stoch_period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < stoch_period // 2:
            continue
        rsi_min = np.min(valid)
        rsi_max = np.max(valid)
        if rsi_max == rsi_min:
            stoch_k_raw[i] = 50.0
        else:
            stoch_k_raw[i] = 100.0 * (rsi_vals[i] - rsi_min) / (rsi_max - rsi_min)

    # Smooth K
    k_smooth_arr = np.full(n, np.nan, dtype=np.float64)
    for i in range(k_smooth - 1, n):
        window = stoch_k_raw[i - k_smooth + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == k_smooth:
            k_smooth_arr[i] = np.mean(valid)

    # D = smooth of K
    d_arr = np.full(n, np.nan, dtype=np.float64)
    for i in range(d_smooth - 1, n):
        window = k_smooth_arr[i - d_smooth + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == d_smooth:
            d_arr[i] = np.mean(valid)

    return k_smooth_arr, d_arr


def _macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MACD line, signal line, and histogram.
    Returns (macd_line, signal_line, histogram).
    """
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(np.where(np.isnan(macd_line), 0.0, macd_line), signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _find_price_pivots(
    prices: np.ndarray, window: int = 5
) -> Tuple[List[int], List[int]]:
    """
    Find pivot highs and lows.
    Returns (pivot_high_indices, pivot_low_indices).
    """
    highs: List[int] = []
    lows: List[int] = []
    n = len(prices)

    for i in range(window, n - window):
        segment = prices[i - window : i + window + 1]
        center = prices[i]
        if center == np.max(segment):
            highs.append(i)
        if center == np.min(segment):
            lows.append(i)

    return highs, lows


def _detect_rsi_divergence(
    closes: np.ndarray, rsi_vals: np.ndarray, lookback: int = 40
) -> str:
    """
    Detect RSI divergence over the last `lookback` bars.
    Returns 'BULLISH_DIV', 'BEARISH_DIV', or 'NONE'.
    """
    if len(closes) < lookback + 10:
        return "NONE"

    price_seg = closes[-lookback:]
    rsi_seg = rsi_vals[-lookback:]

    if np.any(np.isnan(rsi_seg)):
        # Remove NaN pairs
        mask = ~np.isnan(rsi_seg)
        price_seg = price_seg[mask]
        rsi_seg = rsi_seg[mask]

    if len(price_seg) < 10:
        return "NONE"

    price_highs, price_lows = _find_price_pivots(price_seg, window=3)
    _, rsi_lows = _find_price_pivots(rsi_seg, window=3)
    rsi_highs, _ = _find_price_pivots(rsi_seg, window=3)

    # Bullish divergence: price makes lower low, RSI makes higher low
    if len(price_lows) >= 2 and len(rsi_lows) >= 2:
        p_low1 = price_seg[price_lows[-2]]
        p_low2 = price_seg[price_lows[-1]]
        r_low1 = rsi_seg[rsi_lows[-2]]
        r_low2 = rsi_seg[rsi_lows[-1]]

        if p_low2 < p_low1 and r_low2 > r_low1:
            return "BULLISH_DIV"

    # Bearish divergence: price makes higher high, RSI makes lower high
    if len(price_highs) >= 2 and len(rsi_highs) >= 2:
        p_hi1 = price_seg[price_highs[-2]]
        p_hi2 = price_seg[price_highs[-1]]
        r_hi1 = rsi_seg[rsi_highs[-2]]
        r_hi2 = rsi_seg[rsi_highs[-1]]

        if p_hi2 > p_hi1 and r_hi2 < r_hi1:
            return "BEARISH_DIV"

    return "NONE"


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────


@dataclass
class MomentumResult:
    stochrsi_signal: str   # "BUY", "SELL", "NEUTRAL"
    macd_signal: str       # "BUY", "SELL", "NEUTRAL"
    rsi_divergence: str    # "BULLISH_DIV", "BEARISH_DIV", "NONE"
    rsi_value: float
    score: float           # 0-20
    details: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────
# Main analysis function
# ──────────────────────────────────────────────────────────


def analyze(candles: List[Dict], direction_hint: str = "LONG") -> MomentumResult:
    """
    Momentum analysis on 15m candles.

    direction_hint: "LONG" or "SHORT" — used to score alignment.
    """
    _neutral = MomentumResult(
        stochrsi_signal="NEUTRAL",
        macd_signal="NEUTRAL",
        rsi_divergence="NONE",
        rsi_value=50.0,
        score=0.0,
        details={"error": "insufficient data"},
    )

    if not candles or len(candles) < 50:
        logger.warning("Layer3: insufficient candles (%d)", len(candles) if candles else 0)
        return _neutral

    try:
        closes = np.array([c["close"] for c in candles], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Layer3: candle parsing error: %s", exc)
        return _neutral

    # ── RSI ───────────────────────────────────────────────
    rsi_vals = _rsi(closes, 14)
    valid_rsi = rsi_vals[~np.isnan(rsi_vals)]
    rsi_current = float(valid_rsi[-1]) if len(valid_rsi) > 0 else 50.0

    # ── StochRSI ──────────────────────────────────────────
    stoch_k, stoch_d = _stochrsi(closes, 14, 14, 3, 3)
    valid_k = stoch_k[~np.isnan(stoch_k)]
    valid_d = stoch_d[~np.isnan(stoch_d)]

    stochrsi_signal = "NEUTRAL"
    stochrsi_k = float(valid_k[-1]) if len(valid_k) > 0 else 50.0
    stochrsi_d = float(valid_d[-1]) if len(valid_d) > 0 else 50.0

    if len(valid_k) >= 2 and len(valid_d) >= 2:
        prev_k = float(valid_k[-2])
        prev_d = float(valid_d[-2])

        # K crosses above D in oversold zone -> BUY
        if prev_k <= prev_d and stochrsi_k > stochrsi_d and stochrsi_k < 50:
            stochrsi_signal = "BUY"
        # K crosses below D in overbought zone -> SELL
        elif prev_k >= prev_d and stochrsi_k < stochrsi_d and stochrsi_k > 50:
            stochrsi_signal = "SELL"
        # Zone-based fallback
        elif stochrsi_k < 20:
            stochrsi_signal = "BUY"
        elif stochrsi_k > 80:
            stochrsi_signal = "SELL"

    # ── MACD ──────────────────────────────────────────────
    macd_line, sig_line, histogram = _macd(closes, 12, 26, 9)
    valid_hist = histogram[~np.isnan(histogram)]

    macd_signal = "NEUTRAL"
    macd_expanding = False

    if len(valid_hist) >= 2:
        curr_hist = float(valid_hist[-1])
        prev_hist = float(valid_hist[-2])

        # Direction
        if curr_hist > 0:
            macd_signal = "BUY"
        elif curr_hist < 0:
            macd_signal = "SELL"

        # Expanding vs contracting
        macd_expanding = abs(curr_hist) > abs(prev_hist)

    # ── RSI Divergence ────────────────────────────────────
    rsi_divergence = _detect_rsi_divergence(closes, rsi_vals, lookback=40)

    # ── Scoring ───────────────────────────────────────────
    is_long = direction_hint.upper() == "LONG"

    score = 0.0

    # StochRSI component (7 pts)
    if stochrsi_signal == "BUY" and is_long:
        score += 7.0
    elif stochrsi_signal == "SELL" and not is_long:
        score += 7.0
    elif stochrsi_signal == "NEUTRAL":
        score += 3.0
    else:
        # Contradicts direction
        score += 0.0

    # MACD component (7 pts)
    if macd_signal == "BUY" and is_long:
        score += 7.0 if macd_expanding else 5.0
    elif macd_signal == "SELL" and not is_long:
        score += 7.0 if macd_expanding else 5.0
    elif macd_signal == "NEUTRAL":
        score += 3.0
    else:
        score += 0.0

    # RSI divergence component (6 pts)
    if rsi_divergence == "BULLISH_DIV" and is_long:
        score += 6.0
    elif rsi_divergence == "BEARISH_DIV" and not is_long:
        score += 6.0
    elif rsi_divergence == "NONE":
        score += 3.0
    else:
        # Divergence contradicts direction
        score += 0.0

    # Penalty: if momentum directly contradicts direction
    contradictions = 0
    if stochrsi_signal == "SELL" and is_long:
        contradictions += 1
    if stochrsi_signal == "BUY" and not is_long:
        contradictions += 1
    if macd_signal == "SELL" and is_long:
        contradictions += 1
    if macd_signal == "BUY" and not is_long:
        contradictions += 1

    if contradictions >= 2:
        score = max(0.0, score - 5.0)

    score = min(20.0, score)

    details = {
        "rsi": round(rsi_current, 2),
        "stochrsi_k": round(stochrsi_k, 2),
        "stochrsi_d": round(stochrsi_d, 2),
        "stochrsi_signal": stochrsi_signal,
        "macd_signal": macd_signal,
        "macd_expanding": macd_expanding,
        "macd_histogram": round(float(valid_hist[-1]), 6) if len(valid_hist) > 0 else None,
        "rsi_divergence": rsi_divergence,
        "direction_hint": direction_hint,
        "contradictions": contradictions,
    }

    return MomentumResult(
        stochrsi_signal=stochrsi_signal,
        macd_signal=macd_signal,
        rsi_divergence=rsi_divergence,
        rsi_value=round(rsi_current, 2),
        score=round(score, 2),
        details=details,
    )
