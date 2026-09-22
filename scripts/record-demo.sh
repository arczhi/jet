#!/usr/bin/env zsh
# Record or convert the README demo GIF.
#
#   Mode 1 (recommended): record with macOS Screen Capture yourself (⌘⇧5),
#   then: make demo VIDEO=/path/to/recording.mov
#
#   Mode 2 (auto): needs Screen Recording permission for the terminal.
#   make demo   → moves the jet window, sends the demo task via the HTTP API,
#                 records with ffmpeg, converts, patches the README.

set -euo pipefail
cd "$(dirname "$0")/.."

DEMO_DIR="${DEMO_DIR:-$HOME/jet-music-demo}"
GIF="docs/demo.gif"
BASE="${JET_URL:-http://127.0.0.1:8765}"
SCREEN_INDEX="${SCREEN_INDEX:-1}"
DUR="${DUR:-150}"

command -v ffmpeg >/dev/null || { echo "ffmpeg missing: brew install ffmpeg"; exit 1; }

convert() {
  local input="$1"
  mkdir -p docs
  echo "[demo] converting $input → $GIF"
  ffmpeg -y -i "$input" \
    -vf "fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
    -an "$GIF" 2>/dev/null
  ls -la "$GIF"
}

patch_readme() {
  grep -q "docs/demo.gif" README.md && return 0
  python3 - <<'PY'
from pathlib import Path
readme = Path("README.md")
text = readme.read_text()
marker = "See [LICENSE](LICENSE).\n"
block = marker + "\n![jet demo](docs/demo.gif)\n"
readme.write_text(text.replace(marker, block, 1))
print("[demo] README patched")
PY
}

if [[ -n "${VIDEO:-}" ]]; then
  convert "$VIDEO"
  patch_readme
  echo "[demo] done → $GIF"
  exit 0
fi

# ---- mode 1: full automation ----------------------------------------------
curl -s -m 5 "$BASE/health" >/dev/null || { echo "jet client not running on $BASE"; exit 1; }
mkdir -p "$DEMO_DIR"
curl -s -X POST "$BASE/api/workspace" -H "content-type: application/json" -d "{\"path\":\"$DEMO_DIR\"}" >/dev/null
curl -s -X POST "$BASE/api/session/new" >/dev/null
curl -s -X POST "$BASE/api/approval-mode" -H "content-type: application/json" -d '{"mode":"auto"}' >/dev/null

# move the jet window to a clean position
osascript -e 'tell application "System Events" to tell (first process whose name contains "python") to set position of window 1 to {60, 40}' 2>/dev/null || true
BOUNDS=$(osascript -e 'tell application "System Events" to tell (first process whose name contains "python") to get {position, size} of window 1' 2>/dev/null)
echo "[demo] window bounds: $BOUNDS"

echo "[demo] recording ${DUR}s of screen ${SCREEN_INDEX} (ffmpeg avfoundation)"
ffmpeg -y -f avfoundation -framerate 30 -i "${SCREEN_INDEX}:none" \
  -vf "fps=30" -pix_fmt yuv420p -t "$DUR" /tmp/jet-demo-recording.mp4 2>/dev/null &
FFPID=$!

sleep 2
python3 - <<'PY'
import json, time, urllib.request

base = __import__("os").environ.get("JET_URL", "http://127.0.0.1:8765")
task = ("Create player.html in this workspace: a self-contained offline music player — "
        "WebAudio chip-tune synthesizer that plays 3 short generated tracks, a playlist, "
        "and play/pause/next controls. No external resources. After creating it, run "
        "`open player.html` once, then reply with a one-paragraph summary and stop.")

def post(path, body=None):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body or {}).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode())

turn = post("/api/turns", {"task": task})
turn_id = turn["turn_id"]
print("[demo] turn", turn_id)
while True:
    time.sleep(2)
    state = json.loads(urllib.request.urlopen(base + "/api/state", timeout=10).read())
    if not state.get("busy"):
        print("[demo] turn finished")
        break
PY

wait "$FFPID" 2>/dev/null
convert /tmp/jet-demo-recording.mp4
patch_readme
echo "[demo] done → $GIF"
