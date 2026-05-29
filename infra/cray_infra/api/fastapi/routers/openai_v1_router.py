"""
OpenAI v1 API Router - Standard OpenAI-compatible endpoints.
These endpoints are exposed directly under /v1/ to match OpenAI API spec.
"""

from vllm.entrypoints.openai.completion.protocol import (
    CompletionRequest,
)

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)

from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session
from cray_infra.one_server.vllm_app_registry import get_vllm_app
from cray_infra.util.get_config import get_config

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import httpx
import logging

logger = logging.getLogger(__name__)

openai_v1_router = APIRouter()


async def _proxy_vllm_inprocess(path: str, params: dict, error_label: str):
    app = get_vllm_app()
    if app is None:
        return None

    if params.get("stream"):

        async def generator():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://vllm", timeout=120.0
            ) as client:
                async with client.stream("POST", path, json=params) as resp:
                    if resp.status_code != 200:
                        error_text = await resp.aread()
                        logger.error(
                            f"vLLM {error_label} error ({resp.status_code}): {error_text!r}"
                        )
                        yield (
                            f'data: {{"error": "Failed to create {error_label}: '
                            f'{error_text.decode()}"}}\n\n'
                        )
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(
            content=generator(), media_type="text/event-stream"
        )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://vllm", timeout=120.0
    ) as client:
        response = await client.post(path, json=params)
        content_type = response.headers.get("Content-Type", "application/json")
        if response.status_code != 200:
            logger.error(
                f"vLLM {error_label} error ({response.status_code}): {response.content!r}"
            )
        return Response(
            content=response.content,
            status_code=response.status_code,
            media_type=content_type,
        )


async def _proxy_vllm_http(session, url: str, params: dict, error_label: str):
    if params.get("stream"):

        async def generator():
            async with session.post(url, json=params) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"vLLM {error_label} error ({resp.status}): {error_text}")
                    yield (
                        f'data: {{"error": "Failed to create {error_label}: '
                        f'{error_text}"}}\n\n'
                    )
                    return
                async for chunk in resp.content.iter_any():
                    yield chunk

        return StreamingResponse(content=generator(), media_type="text/event-stream")

    async with session.post(url, json=params) as resp:
        body = await resp.read()
        content_type = resp.headers.get("Content-Type", "application/json")
        if resp.status != 200:
            logger.error(f"vLLM {error_label} error ({resp.status}): {body!r}")
        return Response(content=body, status_code=resp.status, media_type=content_type)


async def _proxy_vllm_post(path: str, url: str, params: dict, error_label: str):
    inprocess_response = await _proxy_vllm_inprocess(path, params, error_label)
    if inprocess_response is not None:
        return inprocess_response

    session = get_global_session()
    return await _proxy_vllm_http(session, url, params, error_label)


async def _proxy_vllm_get(path: str, url: str, error_label: str):
    app = get_vllm_app()
    if app is not None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://vllm", timeout=120.0
        ) as client:
            response = await client.get(path)
            if response.status_code == 200:
                return response.json()
            return JSONResponse(
                content={"error": f"Failed to fetch {error_label}: {response.status_code}"},
                status_code=response.status_code,
            )

    session = get_global_session()
    async with session.get(url) as resp:
        if resp.status == 200:
            return await resp.json()
        return JSONResponse(
            content={"error": f"Failed to fetch {error_label}: {resp.status}"},
            status_code=resp.status,
        )


@openai_v1_router.get("/models")
async def list_models():
    """List available models - proxy to vLLM server."""
    config = get_config()
    try:
        return await _proxy_vllm_get(
            "/v1/models",
            config["vllm_api_url"] + "/v1/models",
            "models",
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@openai_v1_router.post("/completions")
async def create_completions(request: CompletionRequest, raw_request: Request):
    """Create completions - proxy to vLLM server."""
    config = get_config()

    logger.info(f"Received completions request: {request.model_dump(exclude_none=True)}")

    allowed_keys = [
        "model",
        "temperature",
        "prompt",
        "max_tokens",
        "stream",
        "tools",
        "tool_choice",
        "response_format",
        "top_p",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
    ]

    params = {
        key: value
        for key, value in request.model_dump(mode='json', exclude_none=True).items()
        if value is not None and key in allowed_keys
    }

    try:
        return await _proxy_vllm_post(
            "/v1/completions",
            config["vllm_api_url"] + "/v1/completions",
            params,
            "completion",
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@openai_v1_router.post("/chat/completions")
async def create_chat_completions(request: ChatCompletionRequest, raw_request: Request):
    """Create chat completions - proxy to vLLM server."""
    config = get_config()

    logger.info(f"Received chat completions request: {request.model_dump(exclude_none=True)}")

    allowed_keys = [
        "model",
        "temperature",
        "messages",
        "max_tokens",
        "stream",
        "tools",
        "tool_choice",
        "response_format",
        "top_p",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
    ]

    params = {
        key: value
        for key, value in request.model_dump(mode='json', exclude_none=True).items()
        if value is not None and key in allowed_keys
    }

    try:
        return await _proxy_vllm_post(
            "/v1/chat/completions",
            config["vllm_api_url"] + "/v1/chat/completions",
            params,
            "chat completion",
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
