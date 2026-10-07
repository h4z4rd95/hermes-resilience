"""Hermes Resilience — unified watchdog entrypoint."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("resilience")

from src.watchdog.watchdog import Watchdog, WatchdogConfig
from src.failover.failover import FallbackChain
from src.context.context import safe_model_switch
from src.telegram_ctl.telegram_ctl import TelegramConfig, single_shot, handle_status


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [resilience] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler("/var/log/hermes-resilience/watchdog.log", mode="a"),
        ],
    )

    cfg = WatchdogConfig()
    wd = Watchdog(cfg)

    # Run one check cycle
    report = wd.run(max_cycles=1)[0]

    # Write report to log dir
    log_dir = Path("/var/log/hermes-resilience")
    log_dir.mkdir(parents=True, exist_ok=True)
    report_file = log_dir / f"report-{int(time.time())}.json"
    report_file.write_text(json.dumps(report, indent=2, default=str))

    # If issues found, notify Telegram
    tg_cfg = TelegramConfig(bot_token=cfg.telebot_token, chat_id=cfg.telechat_id)
    if not report["overall_ok"] and tg_cfg.bot_token:
        status_text = single_shot(tg_cfg, handle_status(tg_cfg))
        if status_text == "sent":
            logger.info("Status notification sent to Telegram")

    # Output
    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["overall_ok"] else 1)


if __name__ == "__main__":
    main()