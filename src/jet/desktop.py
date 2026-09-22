"""Native macOS client: local HTTP service + a real window.

``jet app`` starts the FastAPI service on a background thread and opens a
``pywebview`` window (WKWebView on macOS) on the main thread. If the GUI cannot
start — no display, missing pywebview — the command says so and falls back to the
default browser instead of failing silently.
"""

from __future__ import annotations

import socket
import threading
import time
import webbrowser
from typing import Any, cast

import uvicorn

from jet.agent.loop import Agent
from jet.agent.session import create_agent
from jet.config import Settings
from jet.errors import JetError
from jet.server.app import create_app
from jet.server.manager import AgentFactory, Approver, EventSink, TurnManager

STARTUP_TIMEOUT_S = 15.0


def build_manager(settings: Settings, agent_factory: AgentFactory | None = None) -> TurnManager:
    if agent_factory is not None:
        return TurnManager(agent_factory, settings)

    def factory(emit: EventSink, approve: Approver) -> Agent:
        return cast(Agent, create_agent(settings, emit=emit, approve=approve))

    return TurnManager(factory, settings)


def pick_port(preferred: int) -> int:
    """Use ``preferred`` when it is a free non-zero port, otherwise pick an ephemeral one."""
    if preferred > 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class DesktopServer:
    def __init__(
        self,
        settings: Settings,
        *,
        port: int = 8765,
        agent_factory: AgentFactory | None = None,
    ):
        self.settings = settings
        self.manager = build_manager(settings, agent_factory)
        self.port = pick_port(port)
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(self.manager),
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                access_log=False,
            )
        )
        self._thread = threading.Thread(target=self.server.run, name="jet-server", daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while not self.server.started:
            if not self._thread.is_alive():
                raise JetError("jet server thread exited during startup")
            if time.monotonic() > deadline:
                raise JetError(f"jet server did not start within {STARTUP_TIMEOUT_S}s")
            time.sleep(0.05)

    def stop(self) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=10)


def _apply_light_titlebar(native_window: Any) -> None:
    """Force the light macOS appearance so the titlebar matches the light client.

    pywebview paints the titlebar with ``NSColor.windowBackgroundColor()``, which
    follows the window's effective appearance — dark when the system is dark.
    The client UI is light, so the window opts out of system dark mode here,
    before it is shown, on the main thread (``before_show`` fires there).
    """
    import AppKit  # type: ignore[import-untyped]

    appearance = AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameAqua)
    if appearance is not None:
        native_window.setAppearance_(appearance)


def run_desktop(settings: Settings, *, port: int = 8765, open_browser: bool = False) -> int:
    server = DesktopServer(settings, port=port)
    server.start()
    try:
        if not open_browser:
            try:
                import webview
            except Exception as exc:  # noqa: BLE001 - report why the GUI is unavailable
                print(f"jet: native window unavailable ({type(exc).__name__}: {exc})")
                print(f"jet: opening {server.url} in your browser instead")
                open_browser = True
            else:
                window = webview.create_window(
                    "jet",
                    server.url,
                    width=1280,
                    height=860,
                    min_size=(980, 640),
                    background_color="#f3f8fd",
                )
                assert window is not None
                window.events.before_show += lambda: _apply_light_titlebar(window.native)
                webview.start()
                return 0
        print(f"jet: serving at {server.url} (Ctrl+C to stop)")
        webbrowser.open(server.url)
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        server.stop()
