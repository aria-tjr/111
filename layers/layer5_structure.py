"""
Layer 5 – Market Structure Analysis.

Calculates linear regression channel, VWAP with bands, liquidity zones,
Fibonacci retracement, and volume profile — all from scratch using numpy.
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


def _linear_regression_channel(
    closes: np.ndarray, period: int = 20
) -> Tuple[float, float, float]:
    """
    Compute linear regression channel over last `period` bars.
    Returns (upper_band, midline, lower_band) for the last bar.
    Uses 2 standard deviations.
    """
    n = len(closes)
    if n < period:
        last = closes[-1]
        return last, last, last

    seg = closes[-period:]
    x = np.arange(period, dtype=np.float64)

    # Linear regression
    x_mean = np.mean(x)
    y_mean = np.mean(seg)
    slope = np.sum((x - x_mean) * (seg - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean

    # Regression line values
    reg_line = slope * x + intercept

    # Residuals and std dev
    residuals = seg - reg_line
    std_dev = np.std(residuals)

    # Last bar values
    last_reg = reg_line[-1]
    upper = last_reg + 2 * std_dev
    lower = last_reg - 2 * std_dev

    return upper, last_reg, lower


def _channel_position(close: float, upper: float, mid: float, lower: float) -> str:
    """Determine where price sits in the regression channel."""
    if close > upper:
        return "OUTSIDE_UPPER"
    elif close < lower:
        return "OUTSIDE_LOWER"
    elif close > mid + (upper - mid) * 0.5:
        return "UPPER"
    elif close < mid - (mid - lower) * 0.5:
        return "LOWER"
    return "MIDDLE"


def _vwap_with_bands(
    candles: List[Dict], std_multiplier: float = 2.0
) -> Tuple[float, float, float]:
    """
    Calculate VWAP with standard deviation bands using all provided candles.
    Returns (vwap, upper_band, lower_band).
    """
    if not candles:
        return 0.0, 0.0, 0.0

    try:
        tp = np.array(
            [(c["high"] + c["low"] + c["close"]) / 3 for c in candles], dtype=np.float64
        )
        vol = np.array([max(c["volume"], 1e-10) for c in candles], dtype=np.float64)

        cum_tp_vol = np.cumsum(tp * vol)
        cum_vol = np.cumsum(vol)
        vwap_arr = cum_tp_vol / cum_vol

        vwap = float(vwap_arr[-1])

        # Std dev of price from VWAP
        variance = np.sum(vol * (tp - vwap) ** 2) / np.sum(vol)
        std_dev = np.sqrt(max(variance, 0))

        upper = vwap + std_multiplier * std_dev
        lower = vwap - std_multiplier * std_dev
        return vwap, upper, lower
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        logger.warning("VWAP error: %s", exc)
        return 0.0, 0.0, 0.0


def _vwap_position(close: float, vwap: float, upper: float, lower: float) -> str:
    """Determine VWAP position."""
    if vwap == 0:
        return "UNKNOWN"

    band_range = upper - lower
    if band_range == 0:
        return "AT" if abs(close - vwap) / vwap < 0.001 else "ABOVE" if close > vwap else "BELOW"

    deviation = (close - vwap) / (band_range / 2)

    if abs(deviation) < 0.1:
        return "AT"
    elif close > vwap:
        return "ABOVE"
    return "BELOW"


def _find_liquidity_zones(
    candles: List[Dict], lookback: int = 50, threshold_pct: float = 0.002
) -> Tuple[List[float], bool]:
    """
    Find clusters of recent highs/lows within threshold_pct of each other.
    Returns (zone_levels, near_zone_bool) where near_zone is True if current
    price is within threshold of any zone.
    """
    if not candles or len(candles) < 10:
        return [], False

    recent = candles[-lookback:]
    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]
    current_close = candles[-1]["close"]

    all_levels = highs + lows

    # Cluster levels
    zones: List[float] = []
    used = [False] * len(all_levels)

    for i, level in enumerate(all_levels):
        if used[i]:
            continue
        cluster = [level]
        for j in range(i + 1, len(all_levels)):
            if not used[j]:
                pct_diff = abs(all_levels[j] - level) / level
                if pct_diff <= threshold_pct:
                    cluster.append(all_levels[j])
                    used[j] = True
        if len(cluster) >= 3:  # At least 3 price touches
            zones.append(np.mean(cluster))

    # Check if current price is near any zone
    near_zone = any(
        abs(current_close - z) / current_close <= threshold_pct * 2
        for z in zones
    )

    return zones, near_zone


def _fibonacci_confluence(candles: List[Dict], lookback: int = 100) -> Tuple[bool, Dict]:
    """
    Calculate Fibonacci retracement levels from last swing high/low.
    Returns (is_at_fibonacci_level, details_dict).
    """
    if not candles or len(candles) < 20:
        return False, {}

    recent = candles[-lookback:]
    highs = np.array([c["high"] for c in recent], dtype=np.float64)
    lows = np.array([c["low"] for c in recent], dtype=np.float64)
    current_close = candles[-1]["close"]

    swing_high = float(np.max(highs))
    swing_low = float(np.min(lows))
    swing_range = swing_high - swing_low

    if swing_range == 0:
        return False, {}

    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    retracement_prices = {
        f"{int(f * 1000) / 10}%": swing_high - f * swing_range for f in fib_levels
    }

    # Check if current price is near any Fibonacci level (within 0.3%)
    fib_threshold = current_close * 0.003
    at_fib = any(
        abs(current_close - price) <= fib_threshold
        for price in retracement_prices.values()
    )

    return at_fib, {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "fib_levels": {k: round(v, 6) for k, v in retracement_prices.items()},
        "current": current_close,
        "at_fib": at_fib,
    }


def _volume_profile(
    candles: List[Dict], num_buckets: int = 50
) -> Tuple[str, Dict]:
    """
    Volume profile: divide price range into buckets, find POC, VAH, VAL.
    Returns (zone_str, details_dict).
    zone_str: "VAH", "POC", "VAL", "BETWEEN"
    """
    if not candles or len(candles) < 10:
        return "BETWEEN", {}

    try:
        highs = np.array([c["high"] for c in candles], dtype=np.float64)
        lows = np.array([c["low"] for c in candles], dtype=np.float64)
        closes = np.array([c["close"] for c in candles], dtype=np.float64)
        volumes = np.array([c["volume"] for c in candles], dtype=np.float64)

        price_min = float(np.min(lows))
        price_max = float(np.max(highs))

        if price_max == price_min:
            return "BETWEEN", {}

        bucket_size = (price_max - price_min) / num_buckets
        vol_profile = np.zeros(num_buckets, dtype=np.float64)

        for i, (h, l, v) in enumerate(zip(highs, lows, volumes)):
            lo_bucket = int((l - price_min) / bucket_size)
            hi_bucket = int((h - price_min) / bucket_size)
            lo_bucket = max(0, min(lo_bucket, num_buckets - 1))
            hi_bucket = max(0, min(hi_bucket, num_buckets - 1))

            if hi_bucket == lo_bucket:
                vol_profile[lo_bucket] += v
            else:
                span = hi_bucket - lo_bucket + 1
                for b in range(lo_bucket, hi_bucket + 1):
                    vol_profile[b] += v / span

        # POC = bucket with most volume
        poc_idx = int(np.argmax(vol_profile))
        poc_price = price_min + (poc_idx + 0.5) * bucket_size

        # Value area: 70% of total volume around POC
        total_vol = np.sum(vol_profile)
        target_vol = total_vol * 0.70

        val_idx = poc_idx
        vah_idx = poc_idx
        accumulated = vol_profile[poc_idx]

        lo = poc_idx - 1
        hi = poc_idx + 1
        while accumulated < target_vol and (lo >= 0 or hi < num_buckets):
            lo_vol = vol_profile[lo] if lo >= 0 else 0
            hi_vol = vol_profile[hi] if hi < num_buckets else 0

            if lo_vol >= hi_vol and lo >= 0:
                accumulated += lo_vol
                val_idx = lo
                lo -= 1
            elif hi < num_buckets:
                accumulated += hi_vol
                vah_idx = hi
                hi += 1
            elif lo >= 0:
                accumulated += lo_vol
                val_idx = lo
                lo -= 1
            else:
                break

        val_price = price_min + (val_idx + 0.5) * bucket_size
        vah_price = price_min + (vah_idx + 0.5) * bucket_size

        current = closes[-1]
        poc_range = bucket_size * 1.5

        if abs(current - poc_price) <= poc_range:
            zone = "POC"
        elif abs(current - vah_price) <= poc_range * 2:
            zone = "VAH"
        elif abs(current - val_price) <= poc_range * 2:
            zone = "VAL"
        else:
            zone = "BETWEEN"

        return zone, {
            "poc_price": round(poc_price, 6),
            "vah_price": round(vah_price, 6),
            "val_price": round(val_price, 6),
            "current_price": round(current, 6),
            "zone": zone,
        }

    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        logger.warning("Volume profile error: %s", exc)
        return "BETWEEN", {}


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────


@dataclass
class StructureResult:
    channel_position: str       # "UPPER", "MIDDLE", "LOWER", "OUTSIDE"
    vwap_position: str          # "ABOVE", "BELOW", "AT"
    near_liquidity_zone: bool
    fibonacci_confluence: bool
    volume_profile_zone: str    # "VAH", "POC", "VAL", "BETWEEN"
    score: float                # 0-15
    details: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────
# Main analysis function
# ──────────────────────────────────────────────────────────


def analyze(candles_15m: List[Dict], candles_1h: List[Dict]) -> StructureResult:
    """
    Market structure analysis.

    candles_15m: 15-minute OHLCV list (primary)
    candles_1h: 1-hour OHLCV list (used for VWAP)
    """
    _neutral = StructureResult(
        channel_position="MIDDLE",
        vwap_position="AT",
        near_liquidity_zone=False,
        fibonacci_confluence=False,
        volume_profile_zone="BETWEEN",
        score=5.0,
        details={"error": "insufficient data"},
    )

    if not candles_15m or len(candles_15m) < 20:
        logger.warning(
            "Layer5: insufficient 15m candles (%d)",
            len(candles_15m) if candles_15m else 0,
        )
        return _neutral

    try:
        closes_15m = np.array([c["close"] for c in candles_15m], dtype=np.float64)
    except (KeyError, TypeError) as exc:
        logger.warning("Layer5: candle parsing error: %s", exc)
        return _neutral

    current_close = float(closes_15m[-1])

    # ── Regression Channel ────────────────────────────────
    upper_ch, mid_ch, lower_ch = _linear_regression_channel(closes_15m, period=20)
    channel_pos = _channel_position(current_close, upper_ch, mid_ch, lower_ch)

    # Normalize "OUTSIDE_*" to "OUTSIDE"
    if channel_pos.startswith("OUTSIDE"):
        channel_pos_clean = "OUTSIDE"
    else:
        channel_pos_clean = channel_pos

    # ── VWAP ─────────────────────────────────────────────
    # Use 1h candles for VWAP if available, else 15m
    vwap_candles = candles_1h if candles_1h and len(candles_1h) >= 5 else candles_15m
    vwap, vwap_upper, vwap_lower = _vwap_with_bands(vwap_candles)
    vwap_pos = _vwap_position(current_close, vwap, vwap_upper, vwap_lower)

    # ── Liquidity Zones ──────────────────────────────────
    _, near_liq = _find_liquidity_zones(candles_15m, lookback=50)

    # ── Fibonacci ────────────────────────────────────────
    at_fib, fib_details = _fibonacci_confluence(candles_15m, lookback=100)

    # ── Volume Profile ───────────────────────────────────
    vol_zone, vp_details = _volume_profile(candles_15m, num_buckets=50)

    # ── Scoring ──────────────────────────────────────────
    score = 0.0

    # High-value zones score higher
    high_value_zone = vol_zone in ("POC", "VAH", "VAL")
    near_vwap = vwap_pos == "AT"
    at_channel_extremes = channel_pos in ("UPPER", "LOWER")

    # Base score for being at a recognizable structure
    if high_value_zone:
        score += 5.0
    elif vol_zone == "BETWEEN":
        score += 2.0

    # Fibonacci confluence adds significant value
    if at_fib:
        score += 5.0

    # VWAP alignment
    if near_vwap:
        score += 2.0
    else:
        score += 1.0

    # Liquidity zone
    if near_liq:
        score += 2.0

    # Channel position
    if at_channel_extremes:
        score += 1.0

    score = min(15.0, score)

    details = {
        "channel": {
            "position": channel_pos_clean,
            "upper": round(upper_ch, 6),
            "mid": round(mid_ch, 6),
            "lower": round(lower_ch, 6),
        },
        "vwap": {
            "value": round(vwap, 6),
            "upper_band": round(vwap_upper, 6),
            "lower_band": round(vwap_lower, 6),
            "position": vwap_pos,
        },
        "liquidity": {"near_zone": near_liq},
        "fibonacci": fib_details,
        "volume_profile": vp_details,
        "score_breakdown": {
            "vol_profile": 5.0 if high_value_zone else 2.0,
            "fibonacci": 5.0 if at_fib else 0.0,
            "vwap": 2.0 if near_vwap else 1.0,
            "liquidity": 2.0 if near_liq else 0.0,
            "channel": 1.0 if at_channel_extremes else 0.0,
        },
    }

    return StructureResult(
        channel_position=channel_pos_clean,
        vwap_position=vwap_pos,
        near_liquidity_zone=near_liq,
        fibonacci_confluence=at_fib,
        volume_profile_zone=vol_zone,
        score=round(score, 2),
        details=details,
    )
