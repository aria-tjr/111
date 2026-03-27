"""
Telegram Bot – sends NEXUS signals and alerts via Bot API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

import config

logger = logging.getLogger(__name__)

TELEGRAM_API_BASE = "https://api.telegram.org/bot"


async def _post_message(
    bot_token: str,
    chat_id: str,
    text: str,
    parse_mode: str = "HTML",
) -> bool:
    """
    Internal: POST a message to Telegram.
    Returns True on success, False on failure.
    """
    url = f"{TELEGRAM_API_BASE}{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return True
                body = await resp.text()
                logger.warning(
                    "Telegram API error %s: %s", resp.status, body[:300]
                )
                return False
    except asyncio.TimeoutError:
        logger.warning("Telegram: request timed out")
        return False
    except aiohttp.ClientError as exc:
        logger.warning("Telegram client error: %s", exc)
        return False
    except Exception as exc:
        logger.warning("Telegram unexpected error: %s", exc)
        return False


def _check_config() -> bool:
    """Return True if bot token and chat ID are configured."""
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured (missing token or chat ID), skipping send")
        return False
    return True


async def send_signal(signal_text: str) -> bool:
    """
    Send a formatted NEXUS signal to Telegram.

    The text is wrapped in <pre> tags for monospace rendering.
    Returns True on success.
    """
    if not _check_config():
        return False

    # Escape any HTML entities that might break the pre block
    escaped = signal_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    message = f"<pre>{escaped}</pre>"

    return await _post_message(
        config.TELEGRAM_BOT_TOKEN,
        config.TELEGRAM_CHAT_ID,
        message,
        parse_mode="HTML",
    )


async def send_alert(message: str) -> bool:
    """
    Send a simple alert/text message to Telegram.

    message: plain text or HTML (not wrapped in <pre>).
    Returns True on success.
    """
    if not _check_config():
        return False

    return await _post_message(
        config.TELEGRAM_BOT_TOKEN,
        config.TELEGRAM_CHAT_ID,
        message,
        parse_mode="HTML",
    )
