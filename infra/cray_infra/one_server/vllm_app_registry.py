"""Registry for the in-process vLLM FastAPI app used by one_server mode."""

import asyncio

from fastapi import FastAPI

_vllm_app: FastAPI | None = None
_vllm_ready = asyncio.Event()


def register_vllm_app(app: FastAPI) -> None:
    global _vllm_app
    _vllm_app = app
    _vllm_ready.set()


def get_vllm_app() -> FastAPI | None:
    return _vllm_app


async def wait_for_vllm_app(timeout: float | None = 120) -> FastAPI:
    try:
        await asyncio.wait_for(_vllm_ready.wait(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError("vLLM app did not register within timeout") from exc

    if _vllm_app is None:
        raise RuntimeError("vLLM app registered event set but app is missing")
    return _vllm_app


def clear_vllm_app() -> None:
    global _vllm_app
    _vllm_app = None
    _vllm_ready.clear()
