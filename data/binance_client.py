"""
Async Binance REST client using aiohttp.
Covers both Spot (api.binance.com) and Futures (fapi.binance.com) endpoints.
No SDK dependency required.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp

import config

logger = logging.getLogger(__name__)

SPOT_BASE = "https://api.binance.com/api/v3/"
FUTURES_BASE = "https://fapi.binance.com/fapi/v1/"


class BinanceClient:
    """Async Binance client. Use as an async context manager or call close() explicitly."""

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=15)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "BinanceClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, base: str, endpoint: str, params: Optional[Dict] = None) -> Any:
        url = base + endpoint
        session = await self._get_session()
        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("Binance %s %s -> HTTP %s: %s", base, endpoint, resp.status, text[:200])
                    return None
                return await resp.json()
        except asyncio.TimeoutError:
            logger.warning("Binance request timed out: %s%s", base, endpoint)
            return None
        except aiohttp.ClientError as exc:
            logger.warning("Binance client error: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Binance unexpected error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Return list of OHLCV dicts with keys:
        open_time, open, high, low, close, volume
        """
        # Futures endpoint for perpetual symbols
        data = await self._get(
            FUTURES_BASE,
            "klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        if not data:
            # Fallback to spot
            data = await self._get(
                SPOT_BASE,
                "klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
            )
        if not data:
            return []

        result: List[Dict[str, Any]] = []
        for k in data:
            try:
                result.append(
                    {
                        "open_time": int(k[0]),
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
                )
            except (IndexError, ValueError, TypeError):
                continue
        return result

    async def get_order_book(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Optional[Dict[str, Any]]:
        """
        Return dict with:
          bids: list of [price, qty] (best first)
          asks: list of [price, qty] (best first)
        """
        data = await self._get(
            FUTURES_BASE,
            "depth",
            params={"symbol": symbol, "limit": limit},
        )
        if not data:
            data = await self._get(
                SPOT_BASE,
                "depth",
                params={"symbol": symbol, "limit": limit},
            )
        if not data:
            return None

        try:
            return {
                "bids": [[float(p), float(q)] for p, q in data.get("bids", [])],
                "asks": [[float(p), float(q)] for p, q in data.get("asks", [])],
            }
        except (TypeError, ValueError):
            return None

    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Return latest funding rate as a float (e.g. 0.0001)."""
        data = await self._get(
            FUTURES_BASE,
            "fundingRate",
            params={"symbol": symbol, "limit": 1},
        )
        if not data:
            return None
        try:
            # Returns a list
            return float(data[-1]["fundingRate"])
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    async def get_open_interest(self, symbol: str) -> Optional[float]:
        """Return current open interest (in contracts)."""
        data = await self._get(
            FUTURES_BASE,
            "openInterest",
            params={"symbol": symbol},
        )
        if not data:
            return None
        try:
            return float(data["openInterest"])
        except (KeyError, TypeError, ValueError):
            return None

    async def get_open_interest_hist(
        self,
        symbol: str,
        period: str = "15m",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Return list of dicts with keys:
          timestamp, sumOpenInterest
        """
        data = await self._get(
            "https://fapi.binance.com/futures/data/",
            "openInterestHist",
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return []
        result = []
        for item in data:
            try:
                result.append(
                    {
                        "timestamp": int(item["timestamp"]),
                        "sumOpenInterest": float(item["sumOpenInterest"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return result

    async def get_long_short_ratio(
        self,
        symbol: str,
        period: str = "15m",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Return list of dicts with keys:
          timestamp, longShortRatio, longAccount, shortAccount
        """
        data = await self._get(
            "https://fapi.binance.com/futures/data/",
            "globalLongShortAccountRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return []
        result = []
        for item in data:
            try:
                result.append(
                    {
                        "timestamp": int(item["timestamp"]),
                        "longShortRatio": float(item["longShortRatio"]),
                        "longAccount": float(item["longAccount"]),
                        "shortAccount": float(item["shortAccount"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return result

    async def get_agg_trades(
        self,
        symbol: str,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Return list of aggregated trades for CVD calculation.
        Each dict has: price, qty, is_buyer_maker, time
        """
        data = await self._get(
            FUTURES_BASE,
            "aggTrades",
            params={"symbol": symbol, "limit": limit},
        )
        if not data:
            data = await self._get(
                SPOT_BASE,
                "aggTrades",
                params={"symbol": symbol, "limit": limit},
            )
        if not data:
            return []
        result = []
        for item in data:
            try:
                result.append(
                    {
                        "price": float(item["p"]),
                        "qty": float(item["q"]),
                        "is_buyer_maker": bool(item["m"]),
                        "time": int(item["T"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return result
