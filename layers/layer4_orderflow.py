"""
Layer 4 – Order Flow Analysis.

Analyzes order book imbalance, CVD, open interest trend,
funding rate, and long/short ratio to detect institutional flow.
All math from scratch using numpy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────


@dataclass
class OrderFlowResult:
    bid_ask_imbalance: float    # -1 to 1, positive = bid heavy
    cvd_signal: str             # "BULLISH", "BEARISH", "NEUTRAL"
    oi_trend: str               # "RISING", "FALLING", "FLAT"
    funding_bias: str           # "LONGS_PAYING", "SHORTS_PAYING", "NEUTRAL"
    crowd_positioning: str      # "LONG_HEAVY", "SHORT_HEAVY", "BALANCED"
    fade_risk: bool             # True if crowd too one-sided
    score: float                # 0-20
    details: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────
# Component calculations
# ──────────────────────────────────────────────────────────


def _order_book_imbalance(order_book: Optional[Dict]) -> float:
    """
    Compute bid/ask imbalance in [-1, 1].
    Positive = more bid volume (bullish pressure).
    """
    if not order_book:
        return 0.0

    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])

    if not bids or not asks:
        return 0.0

    try:
        bid_qty = sum(float(b[1]) for b in bids)
        ask_qty = sum(float(a[1]) for a in asks)
        total = bid_qty + ask_qty
        if total == 0:
            return 0.0
        return (bid_qty - ask_qty) / total
    except (IndexError, TypeError, ValueError) as exc:
        logger.warning("Order book imbalance error: %s", exc)
        return 0.0


def _cvd_signal(agg_trades: Optional[List[Dict]]) -> str:
    """
    Cumulative Volume Delta from aggregated trades.
    is_buyer_maker=True means the buyer was the maker (sell-side aggressor -> bearish).
    Returns "BULLISH", "BEARISH", or "NEUTRAL".
    """
    if not agg_trades:
        return "NEUTRAL"

    try:
        buy_vol = 0.0
        sell_vol = 0.0
        for t in agg_trades:
            qty = float(t.get("qty", 0))
            is_buyer_maker = bool(t.get("is_buyer_maker", False))
            if is_buyer_maker:
                # Market sell order hit passive bid -> sell aggression
                sell_vol += qty
            else:
                # Market buy order hit passive ask -> buy aggression
                buy_vol += qty

        cvd = buy_vol - sell_vol
        total = buy_vol + sell_vol
        if total == 0:
            return "NEUTRAL"

        ratio = cvd / total
        if ratio > 0.05:
            return "BULLISH"
        elif ratio < -0.05:
            return "BEARISH"
        return "NEUTRAL"
    except (TypeError, ValueError) as exc:
        logger.warning("CVD calculation error: %s", exc)
        return "NEUTRAL"


def _oi_trend(oi_hist: Optional[List[Dict]]) -> str:
    """
    Open interest trend from history.
    Returns "RISING", "FALLING", or "FLAT".
    """
    if not oi_hist or len(oi_hist) < 6:
        return "FLAT"

    try:
        values = [float(item["sumOpenInterest"]) for item in oi_hist]
        arr = np.array(values, dtype=np.float64)

        # Compare last 5 vs previous 5
        recent = np.mean(arr[-5:])
        prev = np.mean(arr[-10:-5]) if len(arr) >= 10 else np.mean(arr[:-5])

        if prev == 0:
            return "FLAT"

        change_pct = (recent - prev) / prev * 100.0

        if change_pct > 0.5:
            return "RISING"
        elif change_pct < -0.5:
            return "FALLING"
        return "FLAT"
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("OI trend error: %s", exc)
        return "FLAT"


def _funding_bias(funding_rate: Optional[float]) -> str:
    """
    Interpret funding rate.
    > +0.01% -> LONGS_PAYING (bearish pressure on price)
    < -0.01% -> SHORTS_PAYING (bullish pressure on price)
    """
    if funding_rate is None:
        return "NEUTRAL"

    try:
        fr = float(funding_rate)
        if fr > 0.0001:
            return "LONGS_PAYING"
        elif fr < -0.0001:
            return "SHORTS_PAYING"
        return "NEUTRAL"
    except (TypeError, ValueError):
        return "NEUTRAL"


def _crowd_positioning(ls_ratio: Optional[List[Dict]]) -> tuple:
    """
    Determine crowd positioning from long/short ratio.
    Returns (positioning_str, fade_risk_bool).
    """
    if not ls_ratio:
        return "BALANCED", False

    try:
        latest = ls_ratio[-1]
        ratio = float(latest.get("longShortRatio", 1.0))

        if ratio > 1.5:
            return "LONG_HEAVY", True   # Too many longs -> fade risk for longs
        elif ratio < 0.7:
            return "SHORT_HEAVY", True  # Too many shorts -> fade risk for shorts
        return "BALANCED", False
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Long/short ratio error: %s", exc)
        return "BALANCED", False


# ──────────────────────────────────────────────────────────
# Main analysis function
# ──────────────────────────────────────────────────────────


def analyze(
    order_book: Optional[Dict],
    agg_trades: Optional[List[Dict]],
    open_interest: Optional[float],
    oi_hist: Optional[List[Dict]],
    funding_rate: Optional[float],
    ls_ratio: Optional[List[Dict]],
    direction_hint: str = "LONG",
) -> OrderFlowResult:
    """
    Order flow analysis.

    Each of 5 components is worth up to 4 points (total 20).
    Scores are based on alignment with direction_hint.
    """
    is_long = direction_hint.upper() == "LONG"

    # Component 1: Order book imbalance
    imbalance = _order_book_imbalance(order_book)
    ob_score = 0.0
    if abs(imbalance) < 0.05:
        ob_score = 2.0  # neutral
    elif (imbalance > 0) == is_long:
        ob_score = 4.0  # aligned
    else:
        ob_score = 0.0  # contradicts

    # Component 2: CVD
    cvd_sig = _cvd_signal(agg_trades)
    cvd_score = 0.0
    if cvd_sig == "NEUTRAL":
        cvd_score = 2.0
    elif (cvd_sig == "BULLISH") == is_long:
        cvd_score = 4.0
    else:
        cvd_score = 0.0

    # Component 3: OI trend
    oi_sig = _oi_trend(oi_hist)
    oi_score = 0.0
    # Rising OI with rising price = trend continuation (bullish for longs)
    # Falling OI = trend weakening
    if oi_sig == "FLAT":
        oi_score = 2.0
    elif oi_sig == "RISING":
        # Rising OI confirms direction (both long and short breakouts)
        oi_score = 3.0
    else:
        oi_score = 1.0

    # Component 4: Funding rate
    funding_sig = _funding_bias(funding_rate)
    funding_score = 0.0
    if funding_sig == "NEUTRAL":
        funding_score = 2.0
    elif funding_sig == "LONGS_PAYING" and not is_long:
        # Positive funding = shorting pressure, good for shorts
        funding_score = 4.0
    elif funding_sig == "SHORTS_PAYING" and is_long:
        # Negative funding = squeeze potential, good for longs
        funding_score = 4.0
    elif funding_sig == "LONGS_PAYING" and is_long:
        # High funding = bearish pressure against our long
        funding_score = 1.0
    else:
        funding_score = 1.0

    # Component 5: Crowd positioning / fade risk
    crowd_pos, fade_risk = _crowd_positioning(ls_ratio)
    crowd_score = 0.0
    if crowd_pos == "BALANCED":
        crowd_score = 2.0
    elif crowd_pos == "LONG_HEAVY" and not is_long:
        # Longs crowded, contrarian short is good
        crowd_score = 4.0
    elif crowd_pos == "SHORT_HEAVY" and is_long:
        # Shorts crowded, contrarian long is good (short squeeze potential)
        crowd_score = 4.0
    elif crowd_pos == "LONG_HEAVY" and is_long:
        # Going long with the crowd = fade risk
        crowd_score = 0.0
    else:
        crowd_score = 0.0

    # Strong contradicting signal penalty
    strong_contra = 0
    if imbalance < -0.2 and is_long:
        strong_contra += 1
    if imbalance > 0.2 and not is_long:
        strong_contra += 1
    if cvd_sig == "BEARISH" and is_long:
        strong_contra += 1
    if cvd_sig == "BULLISH" and not is_long:
        strong_contra += 1

    raw_score = ob_score + cvd_score + oi_score + funding_score + crowd_score
    if strong_contra >= 2:
        raw_score = max(0.0, raw_score - 4.0)

    score = min(20.0, raw_score)

    details = {
        "bid_ask_imbalance": round(imbalance, 4),
        "ob_score": ob_score,
        "cvd_signal": cvd_sig,
        "cvd_score": cvd_score,
        "oi_trend": oi_sig,
        "oi_score": oi_score,
        "funding_rate": funding_rate,
        "funding_bias": funding_sig,
        "funding_score": funding_score,
        "crowd_positioning": crowd_pos,
        "fade_risk": fade_risk,
        "crowd_score": crowd_score,
        "strong_contradictions": strong_contra,
        "direction_hint": direction_hint,
    }

    return OrderFlowResult(
        bid_ask_imbalance=round(imbalance, 4),
        cvd_signal=cvd_sig,
        oi_trend=oi_sig,
        funding_bias=funding_sig,
        crowd_positioning=crowd_pos,
        fade_risk=fade_risk,
        score=round(score, 2),
        details=details,
    )
