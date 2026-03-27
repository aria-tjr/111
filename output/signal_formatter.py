"""
Signal Formatter – produces ASCII box output for NEXUS signals.
"""

from __future__ import annotations

from typing import Any

from scoring.nexus_score import NexusScore


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

_BOX_WIDTH = 50  # inner width (between vertical bars)


def _line(text: str = "") -> str:
    """Pad a line to box width."""
    padded = f"  {text}"
    return f"║{padded:<{_BOX_WIDTH}}║"


def _divider() -> str:
    return "╠" + "═" * _BOX_WIDTH + "╣"


def _top() -> str:
    return "╔" + "═" * _BOX_WIDTH + "╗"


def _bottom() -> str:
    return "╚" + "═" * _BOX_WIDTH + "╝"


def _check(passing: bool) -> str:
    return "✓" if passing else "✗"


def _fmt_price(price: float) -> str:
    """Format price intelligently based on magnitude."""
    if price == 0:
        return "N/A"
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.4f}%"


# ──────────────────────────────────────────────────────────
# Main format functions
# ──────────────────────────────────────────────────────────


def format_signal(
    symbol: str,
    timeframe: str,
    score: NexusScore,
    l1: Any,
    l2: Any,
    l3: Any,
    l4: Any,
    l5: Any,
    l6: Any,
) -> str:
    """
    Produce a full NEXUS signal box.

    Returns a multi-line ASCII string suitable for console or Telegram <pre> blocks.
    """
    lines = []

    lines.append(_top())
    # Title
    title = f"NEXUS SIGNAL — {symbol} {timeframe}"
    lines.append(_line(title))
    lines.append(_divider())

    # Score + grade
    lines.append(_line(f"Score:  {score.total:.0f}/100    Grade: {score.grade}"))
    lines.append(_line(f"Direction: {score.direction:<8}  Leverage: {score.leverage}x"))
    lines.append(_divider())

    # Layer details
    # L1 Regime
    l1_pass = l1.is_trending and l1.volatility_state in ("normal", "low")
    l1_text = f"L1 Regime:    {_check(l1_pass)} {('TRENDING' if l1.is_trending else 'RANGING'):<9} ADX:{l1.adx:.1f}"
    lines.append(_line(l1_text))

    # L2 Trend
    l2_pass = l2.aligned_count >= 2
    l2_text = f"L2 Trend:     {_check(l2_pass)} {l2.aligned_count}/3 aligned {l2.direction}"
    lines.append(_line(l2_text))

    # L3 Momentum
    l3_pass = l3.macd_signal in ("BUY", "SELL") and l3.stochrsi_signal in ("BUY", "SELL")
    l3_detail = f"MACD:{l3.macd_signal} StochRSI:{l3.stochrsi_signal}"
    l3_text = f"L3 Momentum:  {_check(l3_pass)} {l3_detail}"
    lines.append(_line(l3_text))

    # L4 Order Flow
    l4_pass = l4.score >= 12
    funding_val = l4.details.get("funding_rate")
    funding_str = _fmt_pct(funding_val) if funding_val is not None else "N/A"
    l4_text = f"L4 OrderFlow: {_check(l4_pass)} OI:{l4.oi_trend:<8} Fund:{funding_str}"
    lines.append(_line(l4_text))

    # L5 Structure
    l5_pass = l5.score >= 8
    l5_detail = f"{l5.volume_profile_zone} VWAP:{l5.vwap_position}"
    if l5.fibonacci_confluence:
        l5_detail += " +Fib"
    l5_text = f"L5 Structure: {_check(l5_pass)} {l5_detail}"
    lines.append(_line(l5_text))

    # L6 Session
    l6_pass = l6.session_quality in ("HIGH", "MEDIUM")
    l6_text = f"L6 Session:   {_check(l6_pass)} {l6.current_session:<10} {l6.session_quality}"
    lines.append(_line(l6_text))

    lines.append(_divider())

    # Entry / SL / TP
    lines.append(_line(f"Entry:  {_fmt_price(score.entry_price):<16} ATR: {_fmt_price(score.atr)}"))
    lines.append(_line(f"SL:     {_fmt_price(score.stop_loss):<16} (1.5x ATR)"))
    lines.append(_line(f"TP1:    {_fmt_price(score.take_profit_1):<16} RR 1:{score.risk_reward:.1f}"))
    lines.append(_line(f"TP2:    {_fmt_price(score.take_profit_2):<16} RR 1:{score.risk_reward * 2:.1f}"))
    lines.append(_line(f"TP3:    {_fmt_price(score.take_profit_3):<16} RR 1:{score.risk_reward * 3:.1f}"))

    lines.append(_bottom())

    return "\n".join(lines)


def format_skip(
    symbol: str,
    timeframe: str,
    score: NexusScore,
    reason: str,
) -> str:
    """
    Produce a compact skip/reject signal box.
    """
    lines = []

    lines.append(_top())
    lines.append(_line(f"NEXUS SKIP — {symbol} {timeframe}"))
    lines.append(_divider())
    lines.append(_line(f"Score:  {score.total:.0f}/100   Grade: {score.grade}"))
    lines.append(_line(f"Reason: {reason}"))
    lines.append(_line())
    lines.append(_line("No trade signal generated."))
    lines.append(_bottom())

    return "\n".join(lines)
