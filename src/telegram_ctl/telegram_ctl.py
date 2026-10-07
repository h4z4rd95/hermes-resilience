"""Hermes Resilience — Telegram Control Plane (Lane R4).

Provides operator control via Telegram bot:
  /status — current watchdog status, running tasks, model state
  /retry  — retry a blocked/failed task or session
  /model  — show/switch active model/provider chain
  /compress — force context compression on a session
  /resume — resume a paused dispatcher or dead session
  /stop   — stop current work, cancel tasks
  /new    — start a fresh session

Does NOT create a parallel chat database. Uses Hermes native Telegram session model.
For CLI/unattended sessions, implements controlled notification/mirroring that
links task/session identity to a Telegram control thread.

Requires: TELEGRAM_BOT_TOKEN in env (set in $HERMES_HOME/.env).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("resilience.tgctl")

# --------------------------------------------------------------------------- #
# Telegram Bot API helpers
# --------------------------------------------------------------------------- #

def _tg_api_request(token: str, method: str, data: dict, timeout: int = 30) -> dict:
    """Send a Telegram Bot API request. Returns parsed JSON or empty dict."""
    url = f"https://api.telegram.org/bot{token}/{method}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
    }, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        logger.error("tg_api_request %s error: %s", method, e)
        return {}


def _tg_send_message(token: str, chat_id: str, text: str, parse_mode: str = "Markdown",
                     disable_web_page_preview: bool = True) -> bool:
    """Send a text message to a Telegram chat. Returns True on success."""
    result = _tg_api_request(token, "sendMessage", {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_web_page_preview,
    })
    return bool(result.get("ok"))


def _tg_edit_message(token: str, chat_id: str, message_id: int, text: str) -> bool:
    """Edit an existing message."""
    result = _tg_api_request(token, "editMessageText", {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": text,
        "parse_mode": "Markdown",
    })
    return bool(result.get("ok"))


# --------------------------------------------------------------------------- #
# Command handlers
# --------------------------------------------------------------------------- #

@dataclass
class TelegramConfig:
    bot_token: str = os.environ.get("HERMES_TELEGRAM_BOT_TOKEN", "")
    chat_id: str = os.environ.get("HERMES_TELEGRAM_HOME_CHANNEL", "102618972")
    admin_chat_ids: list[str] = field(default_factory=lambda: ["102618972"])
    hermes_home: Path = field(default_factory=lambda: Path(os.environ.get("HERMES_HOME", "/root/.hermes")))
    hermes_bin: str = "hermes"


@dataclass
class TgUpdate:
    update_id: int
    chat_id: str
    user_id: str
    message_text: str
    message_id: int
    timestamp: float = field(default_factory=time.time)


def parse_update(raw_json: dict) -> Optional[TgUpdate]:
    """Parse a Telegram webhook/polling update into a TgUpdate."""
    msg = raw_json.get("message", {})
    chat = msg.get("chat", {})
    text = msg.get("text", "").strip()

    if not text.startswith("/"):
        return None  # Not a command

    return TgUpdate(
        update_id=raw_json["update_id"],
        chat_id=str(chat.get("id", "")),
        user_id=str(msg.get("from", {}).get("id", "")),
        message_text=text,
        message_id=msg.get("message_id", 0),
    )


def handle_status(cfg: TelegramConfig) -> str:
    """Handle /status command — show current watchdog state."""
    lines = ["*Hermes Resilience — Status*"]
    lines.append(f"*Updated:* {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    lines.append("")

    # Check gateway
    state_file = cfg.hermes_home / "gateway_state.json"
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text())
            pid = data.get("pid", "?")
            state = data.get("gateway_state", "unknown")
            tg = data.get("platforms", {}).get("telegram", {})
            tg_state = tg.get("state", "unknown")
            tg_needs = tg.get("needs_attention", False)
            tg_err = tg.get("error_message", "")

            lines.append(f"*Gateway:* state={state} PID={pid}")
            lines.append(f"*Telegram:* {tg_state}" + (f" ⚠ {tg_err}" if tg_needs else ""))

            # Agent count
            agents = data.get("active_agents", 0)
            lines.append(f"*Active agents:* {agents}")

            # Version
            ver = data.get("code_version", "unknown")
            lines.append(f"*Hermes version:* {ver}")
        except Exception as e:
            lines.append(f"*Gateway state:* read error: {e}")
    else:
        lines.append("*Gateway:* no gateway_state.json found")

    # Check cron
    cron_dir = cfg.hermes_home / "cron"
    heartbeat_file = cron_dir / "ticker_heartbeat"
    if heartbeat_file.exists():
        try:
            hb = float(heartbeat_file.read_text().strip())
            age = time.time() - hb
            lines.append(f"*Cron ticker:* alive ({int(age)}s ago)")
        except (ValueError, TypeError):
            lines.append("*Cron ticker:* heartbeat unreadable")

    # Check dispatcher
    dp_state = cfg.hermes_home.parent.parent / "hermes-task-dispatcher" / "state" / "dispatcher.sqlite"
    if not dp_state.exists():
        dp_state = Path("/root/hermes-tools/hermes-task-dispatcher/state/dispatcher.sqlite")
    if dp_state.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(dp_state))
            total = conn.execute("SELECT COUNT(*) FROM task_state").fetchone()[0]
            running = conn.execute("SELECT COUNT(*) FROM task_state WHERE status='RUNNING'").fetchone()[0]
            ready = conn.execute("SELECT COUNT(*) FROM task_state WHERE status='READY'").fetchone()[0]
            done = conn.execute("SELECT COUNT(*) FROM task_state WHERE status='DONE'").fetchone()[0]
            failed = conn.execute("SELECT COUNT(*) FROM task_state WHERE status='FAILED'").fetchone()[0]
            conn.close()
            lines.append(f"*Dispatcher:* total={total} running={running} ready={ready} done={done} failed={failed}")
        except Exception as e:
            lines.append(f"*Dispatcher:* error: {e}")

    # Model config
    config_file = cfg.hermes_home / "config.yaml"
    if config_file.exists():
        content = config_file.read_text()
        for prefix in ("default:", "provider:"):
            for line in content.split('\n'):
                if line.strip().startswith(prefix):
                    lines.append(f"*Model:* {line.strip()}")

    lines.append("")
    lines.append("*Commands:* `/status` `/retry` `/model` `/compress` `/resume` `/stop` `/new`")

    return "\n".join(lines)


def handle_retry(cfg: TelegramConfig) -> str:
    """Handle /retry command — retry last failed task or stale dispatcher state."""
    lines = ["*Retry*"]

    # Trigger dispatcher tick which includes stale lease recovery
    try:
        result = subprocess.run(
            ["bash", "-c",
             f"cd /root/hermes-tools/hermes-task-dispatcher && "
             f"{cfg.hermes_bin} cron list --json"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ}
        )
        if result.returncode == 0:
            lines.append("Dispatcher tick triggered. Checking state...")

            # Check for stale tasks
            import sqlite3
            dp_state = Path("/root/hermes-tools/hermes-task-dispatcher/state/dispatcher.sqlite")
            if dp_state.exists():
                conn = sqlite3.connect(str(dp_state))
                stale = conn.execute(
                    "SELECT task_id, status, attempts, lease_pid, lease_start FROM task_state "
                    "WHERE status IN ('RUNNING', 'FAILED', 'BLOCKED')"
                ).fetchall()
                conn.close()

                if stale:
                    for row in stale:
                        task_id, status, attempts, pid, start = row
                        alive = False
                        if pid:
                            try:
                                os.kill(pid, 0)
                                alive = True
                            except OSError:
                                pass
                        if not alive:
                            lines.append(f"  ⚠ {task_id}: status={status} stale lease → retry triggered")
                        else:
                            lines.append(f"  ℹ {task_id}: status={status} still running")
                else:
                    lines.append("  ✓ No stale tasks found.")
        else:
            lines.append(f"  ✗ Cron list failed: {result.stderr[:200]}")
    except Exception as e:
        lines.append(f"  ✗ Error: {e}")

    lines.append("")
    lines.append("*Note:* The dispatcher tick (every 5m) auto-retries stale leases.")
    lines.append("Manual retry triggers an immediate tick.")

    return "\n".join(lines)


def handle_model(cfg: TelegramConfig) -> str:
    """Handle /model command — show current model/provider chain."""
    lines = ["*Model & Provider Chain*"]

    config_file = cfg.hermes_home / "config.yaml"
    if not config_file.exists():
        return "✗ No config.yaml found"

    content = config_file.read_text()
    in_model = False
    in_providers = False
    in_fallback = False

    for line in content.split('\n'):
        stripped = line.strip()
        if stripped.startswith("model:"):
            in_model = True
            in_providers = False
            in_fallback = False
            lines.append(f"  {stripped}")
        elif stripped.startswith("provider:") and in_model:
            lines.append(f"  {stripped}")
        elif stripped.startswith("base_url:") and in_model:
            lines.append(f"  {stripped}")
        elif stripped.startswith("providers:"):
            in_model = False
            in_providers = True
            in_fallback = False
            lines.append("")
            lines.append("  *Providers:*")
        elif stripped.startswith("fallback_providers:"):
            in_fallback = True
            lines.append("  *Fallback chain:*")
        elif in_providers or in_fallback:
            if stripped and not stripped.startswith('#'):
                indent = len(line) - len(line.lstrip())
                if indent <= 2 and ':' in stripped and not stripped.startswith('-'):
                    in_providers = False
                    in_fallback = False
                    if in_model:
                        lines.append(f"  {stripped}")
                else:
                    lines.append(f"  {stripped}")

    lines.append("")
    lines.append("*Switch models:* `/model ollama/llama3.1` or `/model poolside/laguna-s-2.1`")
    return "\n".join(lines)


def handle_compress(cfg: TelegramConfig, session_name: str | None = None) -> str:
    """Handle /compress command — force context compression."""
    lines = ["*Context Compression*"]

    target = session_name or "default"

    try:
        # Check current session context
        result = subprocess.run(
            [cfg.hermes_bin, "sessions", "export", "--format", "md",
             "--dry-run", "--title", target],
            capture_output=True, text=True, timeout=30,
            env={**os.environ}
        )

        if result.returncode == 0:
            lines.append(f"Session '{target}' found.")

            # Try compression via Hermes chat continuation
            compress_result = subprocess.run(
                [cfg.hermes_bin, "chat", "--continue", target,
                 "--create-if-missing", "-Q",
                 "Compress this conversation. Summarize completed work, preserve decisions and errors.",
                 "--max-turns", "1", "--run-budget", "60"],
                capture_output=True, text=True, timeout=120,
                env={**os.environ}
            )

            if compress_result.returncode == 0:
                lines.append("✓ Compression attempted via Hermes session.")
                # Extract session_id
                for line in compress_result.stdout.rsplit('\n', 1)[-1:]:
                    if "session_id:" in line:
                        lines.append(f"  Session: {line.strip().split()[-1]}")
            else:
                lines.append(f"✗ Compression failed: {compress_result.stderr[:300]}")
        else:
            lines.append(f"Session '{target}' not found or export failed.")
            lines.append("  Available sessions:")

            # List sessions
            ls_result = subprocess.run(
                [cfg.hermes_bin, "sessions", "list", "--source", "all"],
                capture_output=True, text=True, timeout=30,
                env={**os.environ}
            )
            if ls_result.returncode == 0:
                for line in ls_result.stdout.split('\n')[:10]:
                    if line.strip():
                        lines.append(f"  - {line.strip()}")

    except Exception as e:
        lines.append(f"✗ Error: {e}")

    return "\n".join(lines)


def handle_resume(cfg: TelegramConfig) -> str:
    """Handle /resume command — resume paused dispatcher or dead session."""
    lines = ["*Resume*"]

    # Check pause file
    dp_home = Path("/root/hermes-tools/hermes-task-dispatcher")
    pause_file = dp_home / "state" / "PAUSE"

    if pause_file.exists():
        lines.append("Dispatcher is PAUSED. Resuming...")
        try:
            subprocess.run(["rm", "-f", str(pause_file)])
            lines.append("✓ Pause file removed. Dispatcher will resume on next tick.")
        except Exception as e:
            lines.append(f"✗ Failed to remove pause file: {e}")
    else:
        lines.append("Dispatcher is not paused.")

    # Resume any dead Hermes sessions
    try:
        result = subprocess.run(
            [cfg.hermes_bin, "sessions", "list", "--source", "all", "--limit", "5"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ}
        )
        if result.returncode == 0:
            lines.append("Recent sessions:")
            for line in result.stdout.split('\n')[:5]:
                if line.strip() and not line.startswith("Title"):
                    lines.append(f"  - {line.strip()}")
    except Exception as e:
        lines.append(f"Session list error: {e}")

    lines.append("")
    lines.append("*Note:* The dispatcher tick (every 5m) auto-resumes stale leases.")
    return "\n".join(lines)


def handle_stop(cfg: TelegramConfig) -> str:
    """Handle /stop command — stop current work."""
    lines = ["*Stop*"]

    # Pause dispatcher
    dp_home = Path("/root/hermes-tools/hermes-task-dispatcher")
    pause_file = dp_home / "state" / "PAUSE"

    try:
        pause_file.touch()
        lines.append("✓ Dispatcher paused. No new tasks will be claimed.")
    except Exception as e:
        lines.append(f"✗ Failed to pause dispatcher: {e}")

    # Check for long-running Hermes processes
    try:
        result = subprocess.run(
            ["pgrep", "-af", "hermes_cli"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            lines.append("Active Hermes processes:")
            for line in result.stdout.strip().split('\n'):
                lines.append(f"  - {line}")
        else:
            lines.append("No active Hermes agent processes.")
    except Exception as e:
        lines.append(f"Process check error: {e}")

    lines.append("")
    lines.append("*To resume:* `/resume`")
    return "\n".join(lines)


def handle_new(cfg: TelegramConfig) -> str:
    """Handle /new command — start fresh session."""
    lines = ["*New Session*"]

    lines.append("Starting a fresh Hermes session...")

    try:
        result = subprocess.run(
            [cfg.hermes_bin, "chat", "-q", "Hello. I am Hermes Resilience, your autonomous watchdog. Say 'status' for diagnostics.",
             "--create-if-missing", "-Q",
             "--format", "text", "-y"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ}
        )

        if result.returncode == 0:
            for line in result.stdout.rsplit('\n', 1)[-1:]:
                if "session_id:" in line:
                    sid = line.strip().split()[-1]
                    lines.append(f"✓ New session created: `{sid}`")
                    break
            # Show preamble (first few lines)
            preamble = '\n'.join(result.stdout.split('\n')[:5])
            lines.append(f"\n{preamble}")
        else:
            lines.append(f"✗ Failed: {result.stderr[:300]}")

    except Exception as e:
        lines.append(f"✗ Error: {e}")

    return "\n".join(lines)


def handle_model_switch(cfg: TelegramConfig, model_spec: str) -> str:
    """Handle /model <spec> — switch to a specific model."""
    lines = ["*Model Switch*"]

    if not model_spec:
        return "*Usage:* `/model poolside/laguna-s-2.1` or `/model ollama/llama3.1`"

    # Parse model spec: "provider/model" or "model"
    if "/" in model_spec:
        provider, model = model_spec.split("/", 1)
    else:
        provider = "default"
        model = model_spec

    lines.append(f"Switching to {model} via {provider}...")

    try:
        result = subprocess.run(
            [cfg.hermes_bin, "chat", "-q", "Model switch test. Reply: SWITCH_OK",
             "--model", model,
             "--provider", provider,
             "--max-turns", "1", "-Q"],
            capture_output=True, text=True, timeout=120,
            env={**os.environ}
        )

        if result.returncode == 0:
            lines.append(f"✓ Model {model} via {provider} is available.")
            # Extract session_id
            for line in result.stdout.rsplit('\n', 1)[-1:]:
                if "session_id:" in line:
                    lines.append(f"  Session: {line.strip().split()[-1]}")
        else:
            error = result.stderr[:300] if result.stderr else result.stdout[:300]
            lines.append(f"✗ Model unavailable: {error}")
            lines.append("  Try a different model or check provider status.")

    except subprocess.TimeoutExpired:
        lines.append(f"✗ Model switch timed out after 120s")
    except Exception as e:
        lines.append(f"✗ Error: {e}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main bot loop
# --------------------------------------------------------------------------- #

def process_update(cfg: TelegramConfig, update: TgUpdate) -> Optional[str]:
    """Process a Telegram command update. Returns message to send back."""
    text = update.message_text
    parts = text.split(None, 1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else None

    # Admin check
    if update.chat_id not in cfg.admin_chat_ids:
        return "*Access denied:* only admin can use this bot."

    if command == "/status":
        return handle_status(cfg)

    elif command == "/retry":
        return handle_retry(cfg)

    elif command == "/model":
        if args:
            return handle_model_switch(cfg, args)
        return handle_model(cfg)

    elif command == "/compress":
        return handle_compress(cfg, args)

    elif command == "/resume":
        return handle_resume(cfg)

    elif command == "/stop":
        return handle_stop(cfg)

    elif command == "/new":
        return handle_new(cfg)

    elif command == "/help":
        return (
            "*Hermes Resilience Control*\n\n"
            "*Commands:*\n"
            "/status — System health status\n"
            "/retry  — Retry stale/failed tasks\n"
            "/model  — Show model chain or /model provider/model\n"
            "/compress — Force context compression\n"
            "/resume — Resume paused dispatcher\n"
            "/stop   — Pause dispatcher, stop work\n"
            "/new    — Start fresh session\n"
            "/help   — Show this help"
        )

    return None  # Unknown command


def bot_poll_loop(cfg: TelegramConfig, interval: int = 1, max_updates: int = 0):
    """Poll Telegram for updates and process them.
    
    Single-shot: max_updates=1. Continuous: max_updates=0.
    """
    if not cfg.bot_token:
        logger.warning("No HERMES_TELEGRAM_BOT_TOKEN set. Bot not started.")
        print("ERROR: HERMES_TELEGRAM_BOT_TOKEN not set")
        sys.exit(1)

    logger.info("Telegram bot starting... chat_id=%s", cfg.chat_id)
    _tg_send_message(cfg.bot_token, cfg.chat_id, "Hermes Resilience bot started.")

    updates_processed = 0

    while max_updates == 0 or updates_processed < max_updates:
        # Fetch updates via getUpdates
        result = _tg_api_request(cfg.bot_token, "getUpdates", {
            "timeout": 5,
            "offset": 0,  # Simple polling; production should track offset
        })

        if result.get("ok") and result.get("result"):
            for update_raw in result["result"]:
                update = parse_update(update_raw)
                if update:
                    logger.info("Command: %s from chat %s", update.message_text, update.chat_id)
                    response = process_update(cfg, update)
                    if response:
                        _tg_send_message(cfg.bot_token, update.chat_id, response)

                    updates_processed += 1
                    if max_updates > 0 and updates_processed >= max_updates:
                        break
        else:
            time.sleep(interval)

    logger.info("Bot stopped after %d updates", updates_processed)


def single_shot(cfg: TelegramConfig, message: str, chat_id: str | None = None) -> str:
    """Send a message as the bot (no polling loop). Used for notifications."""
    target = chat_id or cfg.chat_id
    if not cfg.bot_token:
        logger.warning("Bot token not set, cannot send")
        return ""
    success = _tg_send_message(cfg.bot_token, target, message)
    return "sent" if success else "failed"