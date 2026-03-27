"""
Auto Scanner – scans multiple symbols concurrently and sends Telegram alerts.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

import config
from data.binance_client import BinanceClient
from layers import (
    layer1_regime,
    layer2_trend,
    layer3_momentum,
    layer4_orderflow,
    layer5_structure,
    layer6_session,
)
from output import signal_formatter, telegram_bot
from scoring.nexus_score import NexusScore, compute

logger = logging.getLogger(__name__)


async def _fetch_all(client: BinanceClient, symbol: str) -> Dict:
    """Fetch all required data for a symbol concurrently."""
    (
        candles_15m,
        candles_1h,
        candles_4h,
        order_book,
        agg_trades,
        funding_rate,
        open_interest,
        oi_hist,
        ls_ratio,
    ) = await asyncio.gather(
        client.get_klines(symbol, "15m", limit=250),
        client.get_klines(symbol, "1h", limit=250),
        client.get_klines(symbol, "4h", limit=250),
        client.get_order_book(symbol, limit=20),
        client.get_agg_trades(symbol, limit=500),
        client.get_funding_rate(symbol),
        client.get_open_interest(symbol),
        client.get_open_interest_hist(symbol, period="15m", limit=20),
        client.get_long_short_ratio(symbol, period="15m", limit=20),
        return_exceptions=False,
    )

    return {
        "candles_15m": candles_15m or [],
        "candles_1h": candles_1h or [],
        "candles_4h": candles_4h or [],
        "order_book": order_book,
        "agg_trades": agg_trades or [],
        "funding_rate": funding_rate,
        "open_interest": open_interest,
        "oi_hist": oi_hist or [],
        "ls_ratio": ls_ratio or [],
    }


async def scan_symbol(symbol: str) -> Dict:
    """
    Run full NEXUS analysis on one symbol.

    Returns a result dict with keys:
      symbol, score (NexusScore or None), formatted_signal, error
    """
    result: Dict = {
        "symbol": symbol,
        "score": None,
        "formatted_signal": None,
        "error": None,
    }

    try:
        async with BinanceClient() as client:
            data = await _fetch_all(client, symbol)

            # Fetch BTC candles for correlation (if not BTC itself)
            if symbol.upper() not in ("BTCUSDT", "BTCPERP"):
                btc_candles = await client.get_klines("BTCUSDT", "15m", limit=100)
            else:
                btc_candles = data["candles_15m"]

        candles_15m = data["candles_15m"]
        candles_1h = data["candles_1h"]
        candles_4h = data["candles_4h"]

        if len(candles_15m) < 50:
            result["error"] = "insufficient 15m data"
            return result

        candles_by_tf = {
            "4h": candles_4h,
            "1h": candles_1h,
            "15m": candles_15m,
        }

        # L1 – Regime (gate)
        l1 = layer1_regime.analyze(candles_15m)

        if not l1.is_trending:
            nexus = compute(
                l1,
                layer2_trend.TrendResult(0, "MIXED", {}, {}, 0.0, {}),
                layer3_momentum.MomentumResult("NEUTRAL", "NEUTRAL", "NONE", 50.0, 0.0, {}),
                layer4_orderflow.OrderFlowResult(0.0, "NEUTRAL", "FLAT", "NEUTRAL", "BALANCED", False, 0.0, {}),
                layer5_structure.StructureResult("MIDDLE", "AT", False, False, "BETWEEN", 5.0, {}),
                layer6_session.SessionResult("DEAD", "LOW", False, 0.0, {}),
                candles_15m,
                l1,
            )
            result["score"] = nexus
            result["formatted_signal"] = signal_formatter.format_skip(
                symbol, config.PRIMARY_TF, nexus, "ADX below threshold — no trending regime"
            )
            return result

        # L2 – Trend
        l2 = layer2_trend.analyze(candles_by_tf)
        direction_hint = l2.direction if l2.direction in ("LONG", "SHORT") else "LONG"

        # L3-L6 in parallel
        l3_task = asyncio.get_event_loop().run_in_executor(
            None, layer3_momentum.analyze, candles_15m, direction_hint
        )
        l5_task = asyncio.get_event_loop().run_in_executor(
            None, layer5_structure.analyze, candles_15m, candles_1h
        )

        l3 = layer3_momentum.analyze(candles_15m, direction_hint)
        l4 = layer4_orderflow.analyze(
            data["order_book"],
            data["agg_trades"],
            data["open_interest"],
            data["oi_hist"],
            data["funding_rate"],
            data["ls_ratio"],
            direction_hint,
        )
        l5 = layer5_structure.analyze(candles_15m, candles_1h)
        l6 = layer6_session.analyze(symbol, candles_15m, btc_candles, direction_hint)

        nexus = compute(l1, l2, l3, l4, l5, l6, candles_15m, l1)

        if nexus.grade == "SKIP" or nexus.direction == "SKIP":
            formatted = signal_formatter.format_skip(
                symbol,
                config.PRIMARY_TF,
                nexus,
                f"Score {nexus.total:.0f} below threshold ({config.MIN_SCORE})",
            )
        else:
            formatted = signal_formatter.format_signal(
                symbol, config.PRIMARY_TF, nexus, l1, l2, l3, l4, l5, l6
            )

        result["score"] = nexus
        result["formatted_signal"] = formatted

    except Exception as exc:
        logger.exception("scan_symbol(%s) error: %s", symbol, exc)
        result["error"] = str(exc)

    return result


async def scan_once() -> List[Dict]:
    """
    Single scan pass over all SCANNER_SYMBOLS.

    Returns list of result dicts (one per symbol).
    """
    logger.info("Starting scan pass for %d symbols...", len(config.SCANNER_SYMBOLS))

    tasks = [scan_symbol(sym) for sym in config.SCANNER_SYMBOLS]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)


def _print_summary_table(results: List[Dict]) -> None:
    """Print a summary table to console."""
    header = f"{'Symbol':<12} {'Score':>6} {'Grade':>6} {'Direction':<10} {'Leverage':>8}"
    print("\n" + "=" * len(header))
    print("  NEXUS SCANNER RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    # Sort by score descending
    sorted_results = sorted(
        results,
        key=lambda r: (r["score"].total if r["score"] else 0),
        reverse=True,
    )

    for r in sorted_results:
        sym = r["symbol"]
        if r["error"]:
            print(f"  {sym:<12} {'ERROR':>6} {'-':>6} {r['error'][:30]}")
            continue
        sc: Optional[NexusScore] = r["score"]
        if sc is None:
            print(f"  {sym:<12} {'N/A':>6} {'-':>6}")
            continue

        marker = " ***" if sc.total >= config.SCANNER_MIN_SCORE else ""
        print(
            f"  {sym:<12} {sc.total:>6.1f} {sc.grade:>6} {sc.direction:<10} {sc.leverage:>6}x{marker}"
        )

    print("=" * len(header) + "\n")


async def run_scanner() -> None:
    """
    Main scanner loop.

    Scans all SCANNER_SYMBOLS concurrently every SCANNER_INTERVAL_SECONDS.
    Sends Telegram alerts for any symbol with score >= SCANNER_MIN_SCORE.
    Loops forever.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("NEXUS Auto-Scanner started. Interval: %ds", config.SCANNER_INTERVAL_SECONDS)

    while True:
        start_time = time.monotonic()

        try:
            results = await scan_once()
            _print_summary_table(results)

            # Alert for high-scoring signals
            alert_tasks = []
            for r in results:
                sc: Optional[NexusScore] = r["score"]
                if sc and sc.total >= config.SCANNER_MIN_SCORE and sc.direction != "SKIP":
                    logger.info(
                        "HIGH SCORE: %s score=%.1f grade=%s direction=%s",
                        r["symbol"], sc.total, sc.grade, sc.direction,
                    )
                    if r["formatted_signal"]:
                        alert_tasks.append(telegram_bot.send_signal(r["formatted_signal"]))

            if alert_tasks:
                await asyncio.gather(*alert_tasks, return_exceptions=True)

        except Exception as exc:
            logger.exception("Scanner loop error: %s", exc)

        elapsed = time.monotonic() - start_time
        sleep_for = max(0.0, config.SCANNER_INTERVAL_SECONDS - elapsed)
        logger.info("Scan complete in %.1fs. Next scan in %.0fs.", elapsed, sleep_for)
        await asyncio.sleep(sleep_for)
