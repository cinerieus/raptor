"""x-source known-values seeding for RESUMED conversations.

The pre-dispatch gate's contract is "discovered values must come from
the prompt or prior tool outputs". In-run, only user text and
successful tool results enter ``known_values`` — the model's own prose
never does. Seeding from history must apply the SAME rule: pre-fix,
``run_with_history`` seeded from every message's TextBlocks with no
role filter, so a value the model merely asserted in a prior run's
prose became "discovered" after persist/resume, laundering
injected/hallucinated assistant text past ``ToolCallBlocked``.
"""

from __future__ import annotations

from core.llm.tool_use import (
    Message,
    StopReason,
    TextBlock,
    ToolCall,
    ToolCallReturned,
    ToolDef,
    ToolResult,
    TurnResponse,
)
from core.llm.tool_use.loop import ToolUseLoop
from core.llm.tool_use.types import ToolCallBlocked

SHA = "deadbeef00112233"


class _FakeProvider:
    def __init__(self, responses: list[TurnResponse]) -> None:
        self._responses = list(responses)

    def supports_tool_use(self) -> bool: return True
    def supports_prompt_caching(self) -> bool: return True
    def supports_parallel_tools(self) -> bool: return True
    def context_window(self) -> int: return 200_000
    def price_per_million(self) -> tuple[float, float]: return (3.0, 15.0)
    def estimate_tokens(self, text: str) -> int: return max(len(text) // 4, 1)

    def compute_cost(self, response: TurnResponse) -> float:
        return 0.0

    def turn(self, messages, tools, *, system, max_tokens, cache_control,
             **provider_specific) -> TurnResponse:
        if not self._responses:
            raise RuntimeError("fake provider exhausted")
        return self._responses.pop(0)


def _use_sha_tool() -> ToolDef:
    return ToolDef(
        name="use_sha",
        description="uses a discovered sha",
        input_schema={
            "type": "object",
            "properties": {
                "sha": {"type": "string", "x-source": "discovered"},
            },
        },
        handler=lambda inp: f"got {inp.get('sha')}",
    )


def _run_resumed(history: list[Message]):
    events: list = []
    fp = _FakeProvider([
        TurnResponse(
            content=[ToolCall(id="c1", name="use_sha",
                              input={"sha": SHA})],
            stop_reason=StopReason.NEEDS_TOOL_CALL,
            input_tokens=1, output_tokens=1,
        ),
        TurnResponse(
            content=[TextBlock(text="done")],
            stop_reason=StopReason.COMPLETE,
            input_tokens=1, output_tokens=1,
        ),
    ])
    loop = ToolUseLoop(fp, [_use_sha_tool()], events=events.append)
    loop.run_with_history(history, "continue the analysis")
    blocked = [e for e in events if isinstance(e, ToolCallBlocked)]
    returned = [e for e in events if isinstance(e, ToolCallReturned)]
    return blocked, returned


class TestResumedSeeding:
    def test_assistant_prose_does_not_seed(self):
        """A value that only ever appeared in assistant text must stay
        undiscovered on resume — same verdict the in-run gate gives."""
        history = [
            Message(role="user", content=[TextBlock(text="analyse this")]),
            Message(role="assistant", content=[
                TextBlock(text=f"I believe the target sha is {SHA}"),
            ]),
        ]
        blocked, returned = _run_resumed(history)
        assert len(blocked) == 1
        assert blocked[0].call.name == "use_sha"
        # The blocked call surfaces only as an is_error placeholder —
        # the handler never ran.
        assert all(r.result.is_error for r in returned)

    def test_user_text_seeds(self):
        history = [
            Message(role="user", content=[
                TextBlock(text=f"the sha to inspect is {SHA}"),
            ]),
            Message(role="assistant", content=[TextBlock(text="ok")]),
        ]
        blocked, returned = _run_resumed(history)
        assert blocked == []
        assert any(SHA in r.result.content for r in returned)

    def test_successful_tool_result_seeds(self):
        history = [
            Message(role="user", content=[TextBlock(text="analyse this")]),
            Message(role="assistant", content=[
                ToolCall(id="c0", name="discover", input={}),
            ]),
            Message(role="user", content=[
                ToolResult(tool_use_id="c0", content=f'{{"sha": "{SHA}"}}'),
            ]),
        ]
        blocked, returned = _run_resumed(history)
        assert blocked == []
        assert any(SHA in r.result.content for r in returned)

    def test_error_tool_result_does_not_seed(self):
        """Error results don't seed in-run; resumed histories match."""
        history = [
            Message(role="user", content=[TextBlock(text="analyse this")]),
            Message(role="assistant", content=[
                ToolCall(id="c0", name="discover", input={}),
            ]),
            Message(role="user", content=[
                ToolResult(tool_use_id="c0",
                           content=f'{{"sha": "{SHA}"}}', is_error=True),
            ]),
        ]
        blocked, returned = _run_resumed(history)
        assert len(blocked) == 1
        assert all(r.result.is_error for r in returned)

    def test_in_run_assistant_prose_still_does_not_seed(self):
        """In-run behaviour is unchanged: the model announcing a value
        in prose does not discover it."""
        events: list = []
        fp = _FakeProvider([
            TurnResponse(
                content=[TextBlock(text=f"the sha must be {SHA}"),
                         ToolCall(id="c1", name="use_sha",
                                  input={"sha": SHA})],
                stop_reason=StopReason.NEEDS_TOOL_CALL,
                input_tokens=1, output_tokens=1,
            ),
            TurnResponse(
                content=[TextBlock(text="done")],
                stop_reason=StopReason.COMPLETE,
                input_tokens=1, output_tokens=1,
            ),
        ])
        loop = ToolUseLoop(fp, [_use_sha_tool()], events=events.append)
        loop.run("go")
        blocked = [e for e in events if isinstance(e, ToolCallBlocked)]
        assert len(blocked) == 1
