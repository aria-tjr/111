"""
AltFins API client using aiohttp.
Provides pattern detection and momentum signals.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

import aiohttp

import config

logger = logging.getLogger(__name__)

ALTFINS_BASE = "https://altfins.com/api/public/v1/"


class AltFinsClient:
    """Async AltFins client."""

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers: Dict[str, str] = {}
            if config.ALTFINS_API_KEY:
                headers["Authorization"] = f"Bearer {config.ALTFINS_API_KEY}"
            timeout = aiohttp.ClientTimeout(total=15)
            self._session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "AltFinsClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def _get(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        url = ALTFINS_BASE + endpoint
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 401:
                    logger.warning("AltFins: unauthorized — check ALTFINS_API_KEY")
                    return None
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("AltFins HTTP %s: %s", resp.status, text[:200])
                    return None
                return await resp.json()
        except asyncio.TimeoutError:
            logger.warning("AltFins request timed out")
            return None
        except aiohttp.ClientError as exc:
            logger.warning("AltFins client error: %s", exc)
            return None
        except Exception as exc:
            logger.warning("AltFins unexpected error: %s", exc)
            return None

    async def get_signals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve pattern detections and momentum signals for a symbol.

        Returns dict with structure:
          {
            "patterns": [...],           # list of detected chart patterns
            "momentum_signal": str,      # "UP", "DOWN", "NEUTRAL"
            "trend_signal": str,         # "BULLISH", "BEARISH", "NEUTRAL"
            "summary_score": float,      # 0-100 if available
            "raw": {...},                # raw API response
          }
        Returns None on failure.
        """
        if not config.ALTFINS_API_KEY:
            logger.debug("AltFins API key not configured, skipping")
            return None

        # Normalise symbol: e.g. "ETHUSDT" -> "ETH/USDT" or just pass as-is
        clean_symbol = symbol.upper()

        data = await self._get("coins", params={"symbol": clean_symbol})
        if data is None:
            return None

        # AltFins can return a list or a dict depending on endpoint version
        coin_data: Optional[Dict] = None
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    sym = item.get("symbol", "").upper().replace("/", "").replace("-", "")
                    if sym == clean_symbol.replace("/", "").replace("-", ""):
                        coin_data = item
                        break
        elif isinstance(data, dict):
            coin_data = data

        if coin_data is None:
            logger.debug("AltFins: no data found for symbol %s", symbol)
            return None

        # Parse patterns
        patterns = []
        raw_patterns = coin_data.get("patterns", coin_data.get("chartPatterns", []))
        if isinstance(raw_patterns, list):
            for p in raw_patterns:
                if isinstance(p, dict):
                    patterns.append(
                        {
                            "name": p.get("name", p.get("pattern", "unknown")),
                            "direction": p.get("direction", p.get("trend", "unknown")),
                            "reliability": p.get("reliability", p.get("score", 0)),
                        }
                    )

        # Parse momentum/trend
        momentum = str(coin_data.get("momentumSignal", coin_data.get("momentum", "NEUTRAL"))).upper()
        trend = str(coin_data.get("trendSignal", coin_data.get("trend", "NEUTRAL"))).upper()

        # Normalize values
        if "UP" in momentum or "BULL" in momentum or "BUY" in momentum:
            momentum_norm = "UP"
        elif "DOWN" in momentum or "BEAR" in momentum or "SELL" in momentum:
            momentum_norm = "DOWN"
        else:
            momentum_norm = "NEUTRAL"

        if "BULL" in trend or "UP" in trend:
            trend_norm = "BULLISH"
        elif "BEAR" in trend or "DOWN" in trend:
            trend_norm = "BEARISH"
        else:
            trend_norm = "NEUTRAL"

        summary_score = None
        for key in ("score", "summaryScore", "summary_score", "overallScore"):
            val = coin_data.get(key)
            if val is not None:
                try:
                    summary_score = float(val)
                    break
                except (TypeError, ValueError):
                    pass

        return {
            "patterns": patterns,
            "momentum_signal": momentum_norm,
            "trend_signal": trend_norm,
            "summary_score": summary_score,
            "raw": coin_data,
        }
