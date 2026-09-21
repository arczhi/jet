"""End-to-end smoke test over real HTTP.

Starts a local OpenAI-compatible + TypeSafe-style HTTP server, points jet at it
with no injected mocks, and runs a full turn. Exercises the same code paths as a
real deployment: judgment JSON contract, SSE streaming, RLCD context assembly,
policy evaluation, verification, and tracing.

Run with: make smoke
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jet.agent.session import create_agent
from jet.config import LLMProfile, load_settings
from jet.core.events import TurnFinished

ANSWER = "Smoke test complete."


class FakeProviderServer(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.endswith("/systemone"):
            data = self._judgments(body).encode()
            content_type = "application/json"
        elif self.path.endswith("/chat/completions"):
            if body.get("response_format", {}).get("type") == "json_object":
                data = self._judgment_completion(body).encode()
                content_type = "application/json"
            else:
                data = self._completion().encode()
                content_type = "text/event-stream"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _answers(self, questions: dict) -> dict[str, dict]:
        answers: dict[str, dict] = {}
        for key, question in questions.items():
            kind = question.get("type")
            if kind == "noul":
                answers[key] = {"type": "noul", "noul": 0.92}
            elif kind == "score":
                answers[key] = {
                    "type": "score",
                    "score": 3.0,
                    "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 1.0},
                    "confidence": 0.9,
                }
            elif kind == "choice":
                options = list(question.get("criteria", {}))
                answers[key] = {
                    "type": "choice",
                    "choice": options[0],
                    "probabilities": {option: 1.0 / len(options) for option in options},
                    "confidence": 0.5,
                }
        return answers

    def _judgments(self, body: dict) -> str:
        return json.dumps(
            {
                "model": "fake-jev",
                "answers": self._answers(body.get("questions", {})),
                "usage": {"input_tokens": 42, "output_tokens": 8},
            }
        )

    def _judgment_completion(self, body: dict) -> str:
        """Serve the laya-mlx judge contract: questions arrive in the user message."""
        user = json.loads(body["messages"][1]["content"])
        return json.dumps(
            {
                "model": "fake-judge",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps({"answers": self._answers(user.get("questions", {}))}),
                        }
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 12},
            }
        )

    def _completion(self) -> str:
        events = [
            {"choices": [{"delta": {"content": "Smoke "}}]},
            {"choices": [{"delta": {"content": "test complete."}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 4},
            },
        ]
        return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


async def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeProviderServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}/v1"
    print(f"[smoke] fake provider on {base}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / "AGENTS.md").write_text("Smoke rule: be quick.\n")
        (workspace / "hello.txt").write_text("hello from smoke\n")
        settings = load_settings(
            workspace=workspace,
            home=root / "home",
            judge_provider="openai_compat",
            judge_openai_base_url=base,
            judge_openai_model="fake-judge",
            llm_profile="default",
            llm_profiles={
                "default": LLMProfile(base_url=base, model="fake-llm", api_key="not-needed")
            },
            approval_mode="auto",
            verifier_provider="judge",
            max_steps=3,
        )
        agent = create_agent(settings)
        events: list[object] = []
        agent.emit = events.append
        try:
            result = await agent.run_turn("Reply with the smoke test sentence.")
            chunk_count = agent.store.count()
            trace_path = agent.trace.path
        finally:
            await agent.aclose()
        server.shutdown()
        server.server_close()

        finished = [event for event in events if isinstance(event, TurnFinished)]
        ok = (
            result.stopped_reason == "done"
            and result.verification is not None
            and result.verification.satisfied
            and ANSWER in result.text
            and bool(finished)
            and chunk_count >= 2
            and trace_path.is_file()
        )
        print(f"[smoke] stopped={result.stopped_reason} steps={result.steps} text={result.text!r}")
        print(f"[smoke] chunks={chunk_count} trace={trace_path}")
        print("[smoke] PASS" if ok else "[smoke] FAIL")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
