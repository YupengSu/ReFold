"""StreamingLitellmModel against a fake Anthropic endpoint (httpx MockTransport): the
streamed reply is rebuilt into what a non-streamed call returns. No network."""

from __future__ import annotations

import json

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from minisweagent.models.litellm_model import LitellmModel

from refold.models import StreamingLitellmModel

USAGE = {"input_tokens": 2000, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 36000}
TOOLS = [{"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}},
         {"type": "tool_use", "id": "toolu_2", "name": "bash", "input": {"command": "pwd"}}]


def _sse(thinking: str) -> bytes:
    ev = [("message_start", {"type": "message_start", "message": {
              "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-8", "content": [],
              "stop_reason": None, "usage": {**USAGE, "output_tokens": 1}}}),
          ("content_block_start", {"type": "content_block_start", "index": 0,
                                   "content_block": {"type": "thinking", "thinking": "", "signature": ""}})]
    if thinking:  # display "summarized"; under the default "omitted" no text arrives
        ev.append(("content_block_delta", {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "thinking_delta", "thinking": thinking}}))
    ev += [("content_block_delta", {"type": "content_block_delta", "index": 0,
                                    "delta": {"type": "signature_delta", "signature": "SIG"}}),
           ("content_block_stop", {"type": "content_block_stop", "index": 0}),
           ("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
           ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                    "delta": {"type": "text_delta", "text": "PROGRESS: look\nSTALE: -"}}),
           ("content_block_stop", {"type": "content_block_stop", "index": 1})]
    for i, t in enumerate(TOOLS, start=2):
        ev += [("content_block_start", {"type": "content_block_start", "index": i, "content_block": {**t, "input": {}}}),
               ("content_block_delta", {"type": "content_block_delta", "index": i,
                                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(t["input"])}}),
               ("content_block_stop", {"type": "content_block_stop", "index": i})]
    ev += [("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                              "usage": {"output_tokens": 1000}}),
           ("message_stop", {"type": "message_stop"})]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in ev).encode()


def _endpoint(thinking: str, sent: list) -> HTTPHandler:
    def handle(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        sent.append(body)
        if body.get("stream"):
            return httpx.Response(200, content=_sse(thinking), headers={"content-type": "text/event-stream"})
        blocks = [{"type": "thinking", "thinking": thinking, "signature": "SIG"},
                  {"type": "text", "text": "PROGRESS: look\nSTALE: -"}, *TOOLS]
        return httpx.Response(200, json={"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-4-8",
                                         "content": blocks, "stop_reason": "tool_use",
                                         "usage": {**USAGE, "output_tokens": 1000}})
    return HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handle)))


@pytest.mark.parametrize("thinking", ["", "I should list the files."])
def test_a_streamed_reply_is_the_reply_a_plain_call_returns(thinking, monkeypatch):
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "1")
    sent: list[dict] = []
    kwargs = {"api_key": "sk-test", "client": _endpoint(thinking, sent), "max_tokens": 16_384,
              "reasoning_effort": "medium", "drop_params": True, "temperature": 0.0, "parallel_tool_calls": True}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "task"}]
    streamed = StreamingLitellmModel(model_name="anthropic/claude-opus-4-8", model_kwargs=kwargs).query(msgs)
    plain = LitellmModel(model_name="anthropic/claude-opus-4-8", model_kwargs=kwargs).query(msgs)
    assert sent[0]["stream"] is True and "stream" not in sent[1]
    assert sent[0]["thinking"] == {"type": "adaptive"} and sent[0]["output_config"] == {"effort": "medium"}
    assert "temperature" not in sent[0]  # Opus 4.8 rejects sampling parameters; litellm drops them
    # the block the next request must carry back, even with no text (display "omitted")
    assert streamed["thinking_blocks"] == plain["thinking_blocks"] == [
        {"type": "thinking", "thinking": thinking, "signature": "SIG"}]
    assert streamed["content"] == plain["content"]
    assert streamed["extra"]["actions"] == plain["extra"]["actions"] == [
        {"command": "ls", "tool_call_id": "toolu_1"}, {"command": "pwd", "tool_call_id": "toolu_2"}]
    assert streamed["extra"]["cost"] == pytest.approx(plain["extra"]["cost"]) == pytest.approx(0.0655)
    usage = streamed["extra"]["response"]["usage"]
    assert (usage["cache_read_input_tokens"], usage["cache_creation_input_tokens"]) == (36000, 2000)
