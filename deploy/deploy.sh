#!/usr/bin/env bash
# Deploy resilience watchdog to server 65.109.204.200
# Usage: bash deploy.sh
#
# What it does:
#   1. rsyncs the hermes-resilience package to /root/hermes-tools/
#   2. Installs systemd timer + service
#   3. Creates log directory
#   4. Verifies the watchdog runs correctly
#
# Prerequisites:
#   - SSH key at ~/.ssh/hermes_desktop_key
#   - rsync installed
#   - Git repo pushed with latest code

set -euo pipefail

SERVER="root@65.109.204.200"
KEY="~/.ssh/hermes_desktop_key"
REMOTE_DIR="/root/hermes-tools/hermes-resilience"
LOG_DIR="/var/log/hermes-resilience"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$SCRIPT_DIR/.."  # hermes-resilience root

echo "=== Phase 1: Verify local repo ==="
if [ ! -f "$LOCAL_DIR/src/watchdog/watchdog.py" ]; then
    echo "ERROR: Local hermes-resilience not found at $LOCAL_DIR"
    exit 1
fi
echo "Local files OK."

echo ""
echo "=== Phase 2: rsync to server ==="
ssh -i $KEY -o StrictHostKeyChecking=no $SERVER "mkdir -p $LOG_DIR"
rsync -avz --delete \
    -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    "$LOCAL_DIR/" \
    "$SERVER:$REMOTE_DIR/"

echo ""
echo "=== Phase 3: Create deploy scripts on server ==="
ssh -i $KEY -o StrictHostKeyChecking=no $SERVER "mkdir -p $LOG_DIR"

# Create systemd service
ssh -i $KEY -o StrictHostKeyChecking=no $SERVER "cat > /etc/systemd/system/hermes-resilience-watchdog.service <<'SVC'
[Unit]
Description=Hermes Resilience Watchdog
After=network-online.target hermes-gateway.service
Wants=network-online.target hermes-gateway.service

[Service]
Type=oneshot
User=root
WorkingDirectory=/root/hermes-tools/hermes-resilience
Environment=HERMES_HOME=/root/.hermes
Environment=DISPATCHER_HOME=/root/hermes-tools/hermes-task-dispatcher
Environment=PATH=/usr/local/bin:/usr/local/lib/hermes-agent/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/usr/local/lib/hermes-agent/venv/bin/python -m src.run_watchdog
StandardOutput=journal
StandardError=journal

# One-shot: run once, systemd timer handles scheduling
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
SVC"

echo "Service unit written."

# Create systemd timer (every 5 minutes)
ssh -i $KEY -o StrictHostKeyChecking=no $SERVER "cat > /etc/systemd/system/hermes-resilience-watchdog.timer <<'TMR'
[Unit]
Description=Run Hermes Resilience Watchdog every 5 minutes

[Timer]
OnBootSec=60
OnUnitActiveSec=300
AccuracySec=30
Persistent=true

[Install]
WantedBy=timers.target
TMR"

echo "Timer unit written."

echo ""
echo "=== Phase 4: Enable and start ==="
ssh -i $KEY -o StrictHostKeyChecking=no $SERVER "systemctl daemon-reload && systemctl enable hermes-resilience-watchdog.timer && systemctl start hermes-resilience-watchdog.timer"

echo ""
echo "=== Phase 5: Verify timer is active ==="
ssh -i $KEY -o StrictHostKeyChecking=no $SERVER "systemctl is-active hermes-resilience-watchdog.timer && echo 'Timer: ACTIVE' || echo 'Timer: INACTIVE'"

echo ""
echo "=== Phase 6: First manual run ==="
ssh -i $KEY -o StrictHostKeyChecking=no $SERVER "cd $REMOTE_DIR && /usr/local/lib/hermes-agent/venv/bin/python -m src.run_watchdog 2>&1 | head -50"

echo ""
echo "=== DONE ==="
echo "Watchdog deployed. Timer fires every 5 minutes."
echo "Logs: journalctl -u hermes-resilience-watchdog.service"
echo "Status: systemctl status hermes-resilience-watchdog.timer"