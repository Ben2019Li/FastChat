from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, List, Optional
import time
import uuid
import re

app = FastAPI()

class ResponsesRequest(BaseModel):
    model: str
    input: Optional[Any] = None


def _extract_subject(prompt: str) -> str:
    """Try to heuristically extract a short subject from the prompt.
    Falls back to a generic description if none found.
    """
    if not prompt:
        return "a small creature"
    # look for "about <subject>" patterns
    m = re.search(r"about (?:a |an )?([^\.\n,]+)", prompt, re.IGNORECASE)
    if m:
        subj = m.group(1).strip()
        # truncate to first 4 words
        subj = " ".join(subj.split()[:4])
        return subj
    # fallback: take first 4 words of prompt
    return " ".join(prompt.split()[:4])


def _make_three_sentence_story(prompt: str) -> str:
    subject = _extract_subject(prompt)
    # create three short sentences using the subject
    # keep neutral pronouns and simple grammar to avoid incorrect gendering
    first = f"In a peaceful grove beneath a silver moon, {subject} discovered a hidden pool that reflected the stars."
    second = f"As {subject.split()[0]} approached the water, the pool began to shimmer and revealed a pathway to a gentle, magical realm."
    third = f"Filled with wonder, {subject.split()[0]} whispered a wish for all who dream to find their own spark of magic, and left footprints that twinkled like stardust."
    return " ".join([first, second, third])


@app.post("/v1/responses")
async def create_response(request: Request):
    """Implements a simple /v1/responses endpoint compatible with the example curl.

    Expected JSON body (minimal): {"model": "gpt-4.1", "input": "..."}

    This implementation does not call any model; it synthesizes a short, deterministic
    three-sentence response derived from the prompt. It's suitable for local testing
    and OpenAI API-compatible endpoint emulation.
    """
    body = await request.json()
    model = body.get("model") or "gpt-4.1"
    input_field = body.get("input")

    # input may be a string or list; normalize to string
    if isinstance(input_field, list):
        # join list elements if present
        try:
            input_text = " ".join([str(x) for x in input_field])
        except Exception:
            input_text = str(input_field)
    else:
        input_text = str(input_field or "")

    generated_text = _make_three_sentence_story(input_text)

    resp_id = "resp_" + uuid.uuid4().hex
    msg_id = "msg_" + uuid.uuid4().hex
    created_at = int(time.time())

    response_payload = {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": f"{model}-2025-04-14",
        "output": [
            {
                "type": "message",
                "id": msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": generated_text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "parallel_tool_calls": False,
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": True,
        "temperature": body.get("temperature", 1.0),
        "text": {"format": {"type": "text"}},
        "tool_choice": "auto",
        "tools": [],
        "top_p": body.get("top_p", 1.0),
        "truncation": "disabled",
        "usage": {
            "input_tokens": len(input_text.split()),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": len(generated_text.split()),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": len(input_text.split()) + len(generated_text.split()),
        },
        "user": None,
        "metadata": {},
    }

    return response_payload


# Optionally add a lightweight healthcheck
@app.get("/v1/health")
async def health():
    return {"status": "ok"}
