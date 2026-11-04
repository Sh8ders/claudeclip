"""Draft editable social post copy from a clip's transcript."""

import asyncio
import json

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..prompts import load_prompt

CAPTION_MODEL = "claude-haiku-5-5"
CAPTION_PROMPT = load_prompt("social_caption")


async def generate_social_caption(api_key: str, clip: dict, provider: str) -> str:
    model = AnthropicModel(CAPTION_MODEL, provider=AnthropicProvider(api_key=api_key))
    agent = Agent(
        model, output_type=str, system_prompt=CAPTION_PROMPT,
        model_settings={"max_tokens": 2048}, retries=1,
    )
    prompt = json.dumps({
        "platform": provider,
        "headline": (clip.get("hook_title") or "")[:255],
        "transcript": (clip.get("text") or "")[:16000],
    }, ensure_ascii=False)
    result = await asyncio.wait_for(agent.run(prompt), timeout=40)
    caption = result.output.strip()
    if not caption or len(caption) > 2200:
        raise ValueError("Invalid generated caption")
    return caption
