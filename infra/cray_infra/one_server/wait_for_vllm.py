from cray_infra.one_server.vllm_app_registry import get_vllm_app, wait_for_vllm_app
from cray_infra.util.get_config import get_config

import asyncio
import time

import aiohttp
import httpx

import logging

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 120


async def wait_for_vllm(timeout: float = DEFAULT_TIMEOUT_SECONDS):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        app = get_vllm_app()
        if app is not None:
            if await _inprocess_health(app) == 200:
                return
        elif await get_vllm_health() == 200:
            return
        await asyncio.sleep(1)

    raise TimeoutError(
        f"vLLM did not become healthy within {timeout} seconds"
    )


async def _inprocess_health(app) -> int:
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://vllm"
        ) as client:
            response = await client.get("/health")
            return response.status_code
    except Exception as exc:
        logger.error(f"Error getting in-process vLLM health: {exc}")
        return 500


async def get_vllm_health():
    app = get_vllm_app()
    if app is not None:
        return await _inprocess_health(app)

    config = get_config()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(config["vllm_api_url"] + "/health") as response:
                return response.status
    except Exception as exc:
        logger.error(f"Error getting vLLM health: {exc}")
        return 500


async def ensure_vllm_app(timeout: float = DEFAULT_TIMEOUT_SECONDS):
    await wait_for_vllm_app(timeout=timeout)
