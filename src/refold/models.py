"""StreamingLitellmModel: mini-swe-agent's LitellmModel, streaming the reply.

Anthropic wants long replies streamed: with thinking on, a step at max_tokens = 64,000 can
generate for longer than a non-streamed HTTP request survives. The stream is rebuilt into
the response a non-streamed call returns, so the rest of the loop (actions, cost, the
trajectory) is unchanged. litellm's rebuild drops a thinking block whose text is empty,
which is every block under Claude's default `display: "omitted"`; without its block the
next request would not carry the reply as the model wrote it, so the blocks are rebuilt
here from the streamed deltas.
"""

from __future__ import annotations

import litellm
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


class StreamingLitellmModel(LitellmModel):
    def _query(self, messages: list[dict], **kwargs):
        try:
            chunks = list(litellm.completion(
                model=self.config.model_name, messages=messages, tools=[BASH_TOOL],
                **(self.config.model_kwargs | kwargs), stream=True, stream_options={"include_usage": True}))
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise
        response = litellm.stream_chunk_builder(chunks, messages=messages)
        blocks = thinking_blocks(chunks)
        if blocks:
            response.choices[0].message.thinking_blocks = blocks
        return response


def thinking_blocks(chunks) -> list[dict]:
    """The reply's thinking blocks from its streamed deltas, in order. litellm emits each
    block whole on the delta that carries its signature (the text so far plus the
    signature), so those deltas are the blocks; the text may be empty."""
    blocks: list[dict] = []
    for chunk in chunks:
        for choice in chunk.choices or []:
            for b in getattr(choice.delta, "thinking_blocks", None) or []:
                b = b if isinstance(b, dict) else b.model_dump()
                if b.get("type") == "redacted_thinking":
                    blocks.append({"type": "redacted_thinking", "data": b.get("data", "")})
                elif b.get("signature"):
                    blocks.append({"type": "thinking", "thinking": b.get("thinking") or "", "signature": b["signature"]})
    return blocks
